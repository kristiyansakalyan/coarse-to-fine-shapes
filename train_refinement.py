import torch

def smoothness_loss(point_cloud, neighbors):
    # point_cloud: Tensor of shape (N, 3) where N is the number of points
    # neighbors: A list of sets, each set contains the indices of neighboring points for each point
    loss = 0.0
    for i in range(len(point_cloud)):
        for j in neighbors[i]:
            loss += torch.norm(point_cloud[i] - point_cloud[j], p=2)**2
    return loss / len(point_cloud)

def symmetry_loss(point_cloud, symmetry_plane='xz'):
    # point_cloud: Tensor of shape (N, 3)
    # symmetry_plane: A string indicating the plane of symmetry ('xy', 'yz', 'xz')
    
    if symmetry_plane == 'xz':
        mirrored = torch.stack([point_cloud[:, 0], -point_cloud[:, 1], point_cloud[:, 2]], dim=1)
    elif symmetry_plane == 'xy':
        mirrored = torch.stack([point_cloud[:, 0], point_cloud[:, 1], -point_cloud[:, 2]], dim=1)
    elif symmetry_plane == 'yz':
        mirrored = torch.stack([-point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]], dim=1)
    
    return torch.mean(torch.norm(point_cloud - mirrored, dim=1)**2)

def composite_loss(point_cloud, neighbors, lambda_smooth, lambda_symmetry):
    l_smooth = smoothness_loss(point_cloud, neighbors)
    l_symmetry = symmetry_loss(point_cloud)
    return lambda_smooth * l_smooth + lambda_symmetry * l_symmetry

