import torch

from pytorch3d.ops import knn_points

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

    return result


def get_nearest_neighbors_batch_pytorch3d_indicies(batch, N):
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

    return result, nn_idx

def generate_non_overlapping_patches(batch, patch_size=512):
    """
    Generate non-overlapping patches from a point cloud batch.

    :param batch: Tensor of size [batch_size, num_points, num_channels]
    :param patch_size: Size of each patch
    :return: A list of 4 tensors, each of size [batch_size, patch_size, num_channels]
    """
    batch_size, num_points, _ = batch.shape

    assert num_points % patch_size == 0
    iterations = num_points // patch_size

    input = batch.clone()

    patches = []
    for i in range(iterations - 1):
        # Use the existing function to get a patch from this section
        patch, indicies = get_nearest_neighbors_batch_pytorch3d_indicies(input, patch_size)

        # Remove the selected points
        # Create a mask with all True values
        mask = torch.ones(input.shape[:2], dtype=torch.bool)

        # Iterate over each batch to update the mask
        for i in range(indicies.shape[0]):
            mask[i, indicies[i]] = False

        # Apply the mask to the input tensor
        # The mask is broadcasted to match the shape of input_tensor
        input = input[mask].view(batch_size, -1, 3)

        patches.append(patch)

    patches.append(input)

    return torch.stack(patches)