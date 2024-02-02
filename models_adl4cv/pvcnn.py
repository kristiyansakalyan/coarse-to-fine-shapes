import torch
import torch.nn as nn
import functools

import numpy as np
from models_adl4cv.pointnet_utils import index_points, square_distance

from modules.se import SE3d
from modules.shared_mlp import SharedMLP
from modules.voxelization import Voxelization
import modules.functional as F


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3, pos_enc=False):
        super(Attention, self).__init__()
        assert in_ch % num_groups == 0
        if pos_enc:
            if D == 3:
                self.pos_enc = nn.Conv3d(3, in_ch, 1)
            elif D == 2:
                self.pos_enc = nn.Conv2d(3, in_ch, 1)
            elif D == 1:
                self.pos_enc = nn.Conv1d(3, in_ch, 1)

        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)

        elif D == 2:
            self.q = nn.Conv2d(in_ch, in_ch, 1)
            self.k = nn.Conv2d(in_ch, in_ch, 1)
            self.v = nn.Conv2d(in_ch, in_ch, 1)

            self.out = nn.Conv2d(in_ch, in_ch, 1)

        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)


        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)


    def forward(self, x, xyz=None):
        B, C = x.shape[:2]
        h = x

        h = x

        # print(f"Attention: x {x.shape}; xyz: {xyz.shape}")

        # Apply positional encodings if provided
        if xyz is not None:
            pos_enc = self.pos_enc(xyz.transpose(1,2))
            # print(f"Attention: x {x.shape}; pos_enc: {pos_enc.shape}")

            h += pos_enc  # Add positional encodings to the input features



        q = self.q(h).reshape(B,C,-1)
        k = self.k(h).reshape(B,C,-1)
        v = self.v(h).reshape(B,C,-1)


        # print(f"After q,k,v: q {q.shape},  k {k.shape},  v {v.shape}")

        qk = torch.matmul(q.permute(0, 2, 1), k) #* (int(C) ** (-0.5))

        # print(f"After qk: qk {qk.shape}")

        w = self.sm(qk)

        # print(f"After w: w {w.shape}")

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B,C,*x.shape[2:])

        # print(f"After h: h {h.shape}")

        h = self.out(h)

        x = h + x

        x = self.nonlin(self.norm(x))

        return x

class PointAttention(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):

        # print(f"Attention | xyz: {xyz.shape}; features: {features.shape}")
        
        xyz = xyz.transpose(1,2)
        features = features.transpose(1,2)

        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features

        # print("After all bs above")
        x = self.fc1(features)
        # print(f"After conv3d: {x.shape}")
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        # print("After qkv")

        # print("KNN_XYZ Shape: ", knn_xyz.shape)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        # print("After pos_enc")
        
        # print(q[:, :, None].shape, k.shape, pos_enc.shape)

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        # print("After fc_gamma")
        
        attn = torch.nn.functional.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # print("After softmax")
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        # print("After einsum")

        res = self.fc2(res) + pre
        # print("Finished attention")
        
        return res.transpose(1,2), attn
    

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0, attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.use_attn = attention

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels) if attention else nn.BatchNorm3d(out_channels, eps=1e-4),
            Swish() if attention else nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels) if attention else nn.BatchNorm3d(out_channels, eps=1e-4),
            Swish() if attention else nn.LeakyReLU(0.1, True),
         ]
        
        # if self.use_attn:
        #     self.attention = PointAttention(out_channels, attention_dim, attention_knn)

        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)

        # print(f"PVConv | voxel_features: {voxel_features.shape}")

        # if self.use_attn:
        #     res, attn = self.attention(coords, voxel_features)

        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
    

def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1, attention=False):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                      with_se=with_se, normalize=normalize, eps=eps)
        for _ in range(num_blocks):
            if isinstance(block, functools.partial):
                layers.append(block(in_channels, out_channels, attention=attention))
            else:
                layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels



class PVCNN(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
                                          out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        return self.classifier(torch.cat(out_features_list, dim=1))

class PVCNNPartSeg(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
                                          out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        
        logits = self.classifier(torch.cat(out_features_list, dim=1))

        return torch.nn.functional.softmax(logits, dim=1) 
    

class PVCNNUp(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, up_ratio=2, attention=False):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.up_ratio = up_ratio
        self.use_attn = attention

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            attention=attention
        )
        self.point_features = nn.ModuleList(layers)
        # layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
        #                                   out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
        #                                   classifier=True, dim=2, width_multiplier=width_multiplier)
        # self.classifer = nn.Sequential(*layers)

        conv_in = num_shapes + channels_point + concat_channels_point

        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=conv_in, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 256),
            Attention(256, 8, D=2) if self.use_attn else Swish(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 128),
            Attention(128, 8, D=2) if self.use_attn else Swish(),
        )


        # self.upsampler = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.Conv1d(64, num_classes, 1),
        # )

        in_c = 128
        if self.up_ratio > 1:
            in_c = in_c + self.up_ratio

        # TODO: Maybe add attention here as well
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 64),
            Swish(),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))

        # print(len(out_features_list))

        # for feature in out_features_list:
            # print(feature.shape)

        out_features_concatenated = torch.cat(out_features_list, dim=1)

        # print(out_features_concatenated.shape)

        new_points = []
        for i in range(self.up_ratio):
            features = self.classifier(out_features_concatenated.unsqueeze(-1))

            # Break the symmetry
            if self.up_ratio != 1:
                one_hot = torch.zeros(20, self.up_ratio, 2048, 1).to("cuda")
                one_hot[:, i, :, :] = 1
                features = torch.cat((features, one_hot), dim=1)

            new_points.append(features)

        new_points_concatenated = torch.cat(new_points, dim=2)
        
        # print(new_points[0].shape, new_points_concatenated.shape)

        # Can't we here concatenate two times the out_features_list and then try to apply the classifier
        # It's exactly the same as in the PU-Net :D
        
        output = self.upsampler(new_points_concatenated)

        return output.squeeze(-1)
    

class PVCNNUpPositionalEncodings(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, up_ratio=2, attention=False):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.up_ratio = up_ratio
        self.use_attn = attention

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            attention=attention
        )
        self.point_features = nn.ModuleList(layers)
        # layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
        #                                   out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
        #                                   classifier=True, dim=2, width_multiplier=width_multiplier)
        # self.classifer = nn.Sequential(*layers)

        attention_feature_layers = []

        for in_ch in [64, 128, 128, 512, 2048]:
            attention_feature_layers.append(
                Attention(in_ch, 8, D=1, pos_enc=True)
            )

        self.attention_features = nn.ModuleList(attention_feature_layers)

        conv_in = num_shapes + channels_point + concat_channels_point

        
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_in, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 256),
        )

        self.attn1 = Attention(256, 8, D=2, pos_enc=True)

        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 128),
        )

        self.attn2 = Attention(128, 8, D=2, pos_enc=True)


        # self.upsampler = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.Conv1d(64, num_classes, 1),
        # )

        in_c = 128
        if self.up_ratio > 1:
            in_c = in_c + self.up_ratio

        # TODO: Maybe add attention here as well
        # TODO: Maybe in_c -> 256 -> 128 -> num_classes
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 64),
            Swish(),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        coords_reshaped = coords.permute(0, 2, 1).unsqueeze(-1)  # Reshape to [B, 3, N, 1] for 2D convolutions


        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, xyz = self.point_features[i]((features, coords))
            features = self.attention_features[i](features, xyz=xyz.transpose(1,2))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))

        # print(len(out_features_list))

        # for feature in out_features_list:
            # print(feature.shape)

        out_features_concatenated = torch.cat(out_features_list, dim=1)

        # print(out_features_concatenated.shape)

        new_points = []
        for i in range(self.up_ratio):
            features = self.classifier1(out_features_concatenated.unsqueeze(-1))
            features = self.attn1(features, xyz=coords_reshaped)

            features = self.classifier2(features)
            features = self.attn2(features, xyz=coords_reshaped)

            # Break the symmetry
            if self.up_ratio != 1:
                one_hot = torch.zeros(20, self.up_ratio, 2048, 1).to("cuda")
                one_hot[:, i, :, :] = 1
                features = torch.cat((features, one_hot), dim=1)

            new_points.append(features)

        new_points_concatenated = torch.cat(new_points, dim=2)
        
        # print(new_points[0].shape, new_points_concatenated.shape)

        # Can't we here concatenate two times the out_features_list and then try to apply the classifier
        # It's exactly the same as in the PU-Net :D
        
        output = self.upsampler(new_points_concatenated)

        return output.squeeze(-1)
    

class PVCNNUpPointAttention(nn.Module):
    blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, up_ratio=2, attention=False, attn_dim=512):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.up_ratio = up_ratio
        self.use_attn = attention

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            attention=attention
        )
        self.point_features = nn.ModuleList(layers)
        # layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
        #                                   out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
        #                                   classifier=True, dim=2, width_multiplier=width_multiplier)
        # self.classifer = nn.Sequential(*layers)

        attention_feature_layers = []

        for in_ch in [64, 128, 128, 512, 2048]:
            attention_feature_layers.append(
                PointAttention(in_ch, attn_dim, 20)
            )

        self.attention_features = nn.ModuleList(attention_feature_layers)

        conv_in = num_shapes + channels_point + concat_channels_point

        
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_in, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 256),
        )

        self.attn1 = Attention(256, 8, D=2, pos_enc=True)

        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.GroupNorm(8, 128),
        )

        self.attn2 = Attention(128, 8, D=2, pos_enc=True)


        # self.upsampler = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.Conv1d(64, num_classes, 1),
        # )

        in_c = 128
        if self.up_ratio > 1:
            in_c = in_c + self.up_ratio

        self.upsampler = nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=(1, 1), stride=(1, 1))

        self.attn3 = Attention(64, 8, D=2, pos_enc=True)

        self.upsampler_last = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)

        coords = features[:, :3, :]
        coords_reshaped = coords.permute(0, 2, 1).unsqueeze(-1)  # Reshape to [B, 3, N, 1] for 2D convolutions


        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, xyz = self.point_features[i]((features, coords))
            features, _ = self.attention_features[i](xyz, features)
            # print(f"After attention: {features.shape}")
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))

        # print(len(out_features_list))

        # for feature in out_features_list:
            # print(feature.shape)

        out_features_concatenated = torch.cat(out_features_list, dim=1)

        # print(out_features_concatenated.shape)

        new_points = []
        for i in range(self.up_ratio):
            features = self.classifier1(out_features_concatenated.unsqueeze(-1))
            features = self.attn1(features, xyz=coords_reshaped)

            features = self.classifier2(features)
            features = self.attn2(features,  xyz=coords_reshaped)

            # Break the symmetry
            if self.up_ratio != 1:
                B, _, N, _ = features.shape 
                one_hot = torch.zeros(B, self.up_ratio, N, 1).to("cuda")
                one_hot[:, i, :, :] = 1
                features = torch.cat((features, one_hot), dim=1)

            new_points.append(features)


        new_points_concatenated = torch.cat(new_points, dim=2)
        
        xyz_new = torch.cat([coords_reshaped] * self.up_ratio, dim=1)


        output = self.upsampler(new_points_concatenated)
        output = self.attn3(output, xyz=xyz_new)
        output = self.upsampler_last(output)

        return output.squeeze(-1)
 