import numpy as np
import os
from torch.utils.data import Dataset
import torch
import json

from visuals.visuals_util import rotate_point_cloud_y

def pc_normalize(point_set):
    """
    Normalize the point cloud.
    Centers the point cloud at the origin and scales it to have a maximum extent of 1.

    Args:
        point_set (np.array): The point cloud data. Shape (N, 3), where N is the number of points.

    Returns:
        np.array: The normalized point cloud data.
    """
    centroid = np.mean(point_set, axis=0)
    point_set = point_set - centroid
    furthest_distance = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
    point_set = point_set / furthest_distance
    return point_set


class PartNormalDataset(Dataset):
    def __init__(self, 
                 root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                 npoints=2048, 
                 split='train', 
                 class_choice=None, 
                 normal_channel=False,
                 normalize_per_shape=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.normalize_per_shape = normalize_per_shape

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if class_choice is not None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        if not self.normalize_per_shape:
            # Compute the global mean and std for normalization across the dataset
            all_points = []
            for _, path in self.datapath:
                data = np.loadtxt(path).astype(np.float32)
                if not normal_channel:
                    all_points.append(data[:, 0:3])
                else:
                    all_points.append(data[:, 0:6])

        all_points = np.concatenate(all_points, axis=0)
        self.global_mean = np.mean(all_points, axis=0)
        self.global_std = np.std(all_points, axis=0)


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)


            if self.normalize_per_shape:
                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            else:
                point_set[:, 0:3] = (point_set[:, 0:3] - self.global_mean) / self.global_std
            
            # Rotate so that we have the same alignment as in the other datasets
            point_set[:, 0:3] = rotate_point_cloud_y(point_set[:, 0:3], 90)


            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
            
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)

