
import torch
import modules.functional as F

def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint, 3)
    idx = F.ball_query(pred, pred, radius, nsample)

    # Grouping operation in PyTorch
    grouped_pred = F.grouping(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= pred.unsqueeze(3)

    # Calculate the uniform loss
    h = 0.03
    dist_square = torch.sum(grouped_pred ** 2, dim=-1)
    dist_square, idx = torch.topk(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # Remove the first one
    dist_square = torch.clamp(dist_square, min=1e-12)
    dist = torch.sqrt(dist_square)
    weight = torch.exp(-dist_square / h ** 2)
    uniform_loss = torch.mean(radius - dist * weight)
    return uniform_loss