import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from datasets.shapenet_data_pc import cate_to_synsetid

class Uniform15KPCPVD(Dataset):
    def __init__(
        self,
        root_dir,
        subdirs,
        tr_sample_size=10000,
        te_sample_size=10000,
        split="train",
        scale=1.0,
        normalize_per_shape=False,
        box_per_shape=False,
        random_subsample=False,
        normalize_std_per_axis=False,
        all_points_mean=None,
        all_points_std=None,
        input_dim=3,
        use_mask=False,
    ):
        """?

        Args:
            root_dir (_type_): _description_
            subdirs (_type_): _description_
            tr_sample_size (int, optional): _description_. Defaults to 10000.
            te_sample_size (int, optional): _description_. Defaults to 10000.
            split (str, optional): _description_. Defaults to "train".
            scale (float, optional): _description_. Defaults to 1.0.
            normalize_per_shape (bool, optional): _description_. Defaults to False.
            box_per_shape (bool, optional): _description_. Defaults to False.
            random_subsample (bool, optional): _description_. Defaults to False.
            normalize_std_per_axis (bool, optional): _description_. Defaults to False.
            all_points_mean (_type_, optional): _description_. Defaults to None.
            all_points_std (_type_, optional): _description_. Defaults to None.
            input_dim (int, optional): _description_. Defaults to 3.
            use_mask (bool, optional): _description_. Defaults to False.
        """
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.use_mask = use_mask
        self.box_per_shape = box_per_shape

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith(".npy"):
                    continue
                all_mids.append(os.path.join(self.split, x[: -len(".npy")]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)

                except Exception as e:
                    print(e)
                    continue

                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_points = self.all_points.transpose(0, 2, 1)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if (
            all_points_mean is not None and all_points_std is not None
        ):  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = (
                    self.all_points.reshape(B, N, -1)
                    .std(axis=1)
                    .reshape(B, 1, input_dim)
                )
            else:
                self.all_points_std = (
                    self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
                )
        elif self.box_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.min(axis=1).reshape(B, 1, input_dim)

            self.all_points_std = self.all_points.max(axis=1).reshape(
                B, 1, input_dim
            ) - self.all_points.min(axis=1).reshape(B, 1, input_dim)

        else:  # normalize across the dataset
            self.all_points_mean = (
                self.all_points.reshape(-1, input_dim)
                .mean(axis=0)
                .reshape(1, 1, input_dim)
            )
            if normalize_std_per_axis:
                self.all_points_std = (
                    self.all_points.reshape(-1, input_dim)
                    .std(axis=0)
                    .reshape(1, 1, input_dim)
                )
            else:
                self.all_points_std = (
                    self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
                )

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.test_points = self.all_points
        if self.box_per_shape:
            self.all_points = self.all_points - 0.5
            
        self.train_points = self.all_points

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print(
            "Min number of points: (train)%d (test)%d"
            % (self.tr_sample_size, self.te_sample_size)
        )
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape or self.box_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        out = {
            "idx": idx,
            "train_points": tr_out,
            "test_points": te_out,
            "mean": m,
            "std": s,
            "cate_idx": cate_idx,
            "sid": sid,
            "mid": mid,
        }

        if self.use_mask:
            # masked = torch.from_numpy(self.mask_transform(self.all_points[idx]))
            # ss = min(masked.shape[0], self.in_tr_sample_size//2)
            # masked = masked[:ss]
            #
            # tr_mask = torch.ones_like(masked)
            # masked = torch.cat([masked, torch.zeros(self.in_tr_sample_size - ss, 3)],dim=0)#F.pad(masked, (self.in_tr_sample_size-masked.shape[0], 0), "constant", 0)
            #
            # tr_mask =  torch.cat([tr_mask, torch.zeros(self.in_tr_sample_size- ss, 3)],dim=0)#F.pad(tr_mask, (self.in_tr_sample_size-tr_mask.shape[0], 0), "constant", 0)
            # out['train_points_masked'] = masked
            # out['train_masks'] = tr_mask
            tr_mask = self.mask_transform(tr_out)
            out["train_masks"] = tr_mask

        return out

class ShapeNet15kPointCloudsPVD(Uniform15KPCPVD):
    def __init__(
        self,
        root_dir="data/ShapeNetCore.v2.PC15k",
        categories=["airplane"],
        tr_sample_size=10000,
        te_sample_size=2048,
        split="train",
        scale=1.0,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        box_per_shape=False,
        random_subsample=False,
        all_points_mean=None,
        all_points_std=None,
        use_mask=False,
    ):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ["train", "test", "val"]
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if "all" in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointCloudsPVD, self).__init__(
            root_dir,
            self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3,
            use_mask=use_mask,
        )