import torch
from utils.visualize import visualize_pointclod_trajectory

data_tensor = torch.load("airplane_pc_data.pth")

visualize_pointclod_trajectory("whatever.gif", data_tensor)