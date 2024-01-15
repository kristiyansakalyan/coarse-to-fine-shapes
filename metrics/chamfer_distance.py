from metrics.ChamferDistancePytorch.chamfer_python import distChamfer
import torch


def chamfer_distance(a, b):
    min_dist_a_to_b, min_dist_b_to_a, _, _ = distChamfer(a, b)
    return (min_dist_a_to_b.mean() + min_dist_b_to_a.mean()) / 2


def np_chamfer_distance(point_cloud1, point_cloud2):
    """
    Compute the Chamfer Distance between two point clouds.

    Parameters:
    - point_cloud1: A tensor of shape (N, P, 3)
    - point_cloud2: A tensor of shape (N, P, 3)

    Returns:
    - A tensor containing the Chamfer Distance for each pair of point clouds.
    """
    (
        N,
        P,
        _,
    ) = point_cloud1.shape  # Assume point_cloud1 and point_cloud2 have the same shape

    # Expand dims to (N, P, 1, 3) for broadcasting
    point_cloud1_expanded = point_cloud1.unsqueeze(2)
    point_cloud2_expanded = point_cloud2.unsqueeze(1)

    # Compute squared distances (N, P, P)
    dists = torch.sum((point_cloud1_expanded - point_cloud2_expanded) ** 2, dim=-1)

    # Minimum along one axis (N, P)
    min_dists1 = torch.min(dists, dim=2)[0]  # (N, P)
    min_dists2 = torch.min(dists, dim=1)[0]  # (N, P)

    # Average minimum distance
    chamfer_dist = torch.mean(min_dists1, dim=1) + torch.mean(min_dists2, dim=1)

    return chamfer_dist
