from pytorch3d.ops import knn_points
import torch


def get_nearest_neighbors_batch_pytorch3d(batch, N):
    """
    For each point cloud in the batch, select a random point and find its N nearest neighbors using PyTorch3D.

    :param batch: Tensor of size [batch_size, num_points, num_channels]
    :param N: Number of nearest neighbors to find
    :return: Tensor of nearest neighbors of size [batch_size, N, num_channels]
    """
    batch_size, num_points, _ = batch.shape

    # Select a random point from each point cloud in the batch
    indices = torch.randint(0, num_points, (batch_size,)).to(batch.device)
    selected_points = batch[torch.arange(batch_size), indices].unsqueeze(1)

    # Use knn from PyTorch3D
    knn_result = knn_points(selected_points, batch, K=N)

    nn_idx = knn_result.idx.squeeze()

    # Add an extra dimension to selected_points to make it [20, 255, 1]
    selected_points_expanded = nn_idx.unsqueeze(-1)

    # Expand selected_points to match the last dimension of input_tensor
    selected_points_expanded = selected_points_expanded.expand(-1, -1, batch.shape[-1])

    # Gather the points from the input_tensor
    result = torch.gather(batch, 1, selected_points_expanded)

    result = torch.cat([result, selected_points], dim=1)

    return result
