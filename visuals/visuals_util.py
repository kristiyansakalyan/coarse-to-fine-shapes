from typing import List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_pvd_output(filename: str) -> np.ndarray:
    """
    Loads the final output pointcloud of the Point-Voxel CNN from a file.
    """
    return torch.load(filename, map_location=torch.device("cpu")).numpy()[-1, :, :]


def rotate_point_cloud_x(point_cloud, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    rotated_points = np.dot(point_cloud, rotation_matrix.T)
    return rotated_points


def rotate_point_cloud_y(point_cloud, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)],
        ]
    )
    rotated_points = np.dot(point_cloud, rotation_matrix.T)
    return rotated_points


def rotate_point_cloud_z(point_cloud, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1],
        ]
    )
    rotated_points = np.dot(point_cloud, rotation_matrix.T)
    return rotated_points


def get_axis_ranges(pc: np.ndarray) -> List[Tuple[float, float]]:
    """Get the axis ranges for the given pointcloud i.e. the min and max values for each axis

    Args:
        pc (np.ndarray): pointcloud

    Returns:
        List[Tuple[float, float]]: axis ranges
    """
    max_range = max(pc[:, i].max() - pc[:, i].min() for i in range(3)) / 2.0

    mid_x, mid_y, mid_z = ((pc[:, i].max() + pc[:, i].min()) / 2.0 for i in range(3))
    return [(m - max_range, m + max_range) for m in [mid_x, mid_y, mid_z]]


def visualize_pointcloud(
    pc: np.ndarray,
    title: str = "",
    show_axis: bool = True,
    axis_ranges: Optional[List[Tuple[float, float]]] = None,
) -> plt.Figure:
    """Visualize the given pointcloud

    Args:
        pc (np.ndarray): pointcloud
        title (str, optional): Diagram title. Defaults to "".
        show_axis (bool, optional): Show axis and background. Defaults to True.
        output (str, optional): Output file, does not save if None. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", facecolor="none")

    pc = rotate_point_cloud_y(pc, -90)
    pc = rotate_point_cloud_x(pc, 90)

    # Depth color mapping
    depth = pc[:, 2]
    depth_colormap = plt.get_cmap("viridis")

    # Plot
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=depth, cmap=depth_colormap)

    # Labels
    if show_axis:
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        ax.set_title(title)

    # Setting the aspect ratio
    if axis_ranges is None:
        axis_ranges = get_axis_ranges(pc)

    ax.set_xlim(*axis_ranges[0])
    ax.set_ylim(*axis_ranges[1])
    ax.set_zlim(*axis_ranges[2])

    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.axis("off")

    ax.view_init(15, 0)

    return fig


def save_fig(fig: plt.Figure, filename: str):
    fig.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True, dpi=300)


def visualize_pointcloud_eval(
    pc1: np.ndarray, pc2: np.ndarray,
    title1: str = "", title2: str = "",
    show_axis: bool = True,
    axis_ranges: Optional[List[Tuple[float, float]]] = None,
) -> plt.Figure:
    """Visualize two given pointclouds side by side.

    Args:
        pc1, pc2 (np.ndarray): Pointclouds to be visualized.
        title1, title2 (str, optional): Titles for the diagrams. Defaults to "".
        show_axis (bool, optional): Show axis and background. Defaults to True.
        axis_ranges (Optional[List[Tuple[float, float]]], optional): Axis ranges. Defaults to None.
    """
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d', 'facecolor': 'none'}, figsize=(12, 6))

    for ax, pc, title in zip(axes, [pc1, pc2], [title1, title2]):
        pc = rotate_point_cloud_y(pc, -90)
        pc = rotate_point_cloud_x(pc, 90)

        # Depth color mapping
        depth = pc[:, 2]
        depth_colormap = plt.get_cmap("viridis")

        # Plot
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=depth, cmap=depth_colormap)

        # Labels and title
        if show_axis:
            ax.set_xlabel("Z")
            ax.set_ylabel("X")
            ax.set_zlabel("Y")
            ax.set_title(title)

        # Setting the aspect ratio
        if axis_ranges is None:
            axis_ranges = get_axis_ranges(pc)

        ax.set_xlim(*axis_ranges[0])
        ax.set_ylim(*axis_ranges[1])
        ax.set_zlim(*axis_ranges[2])

        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        ax.view_init(15, 0)

    plt.tight_layout()
    return fig

def visualize_pointcloud_eval_three(
    pc1: np.ndarray, pc2: np.ndarray, pc3: np.ndarray,
    title1: str = "", title2: str = "", title3: str = "",
    show_axis: bool = True,
    axis_ranges: Optional[List[Tuple[float, float]]] = None,
) -> plt.Figure:
    """Visualize two given pointclouds side by side.

    Args:
        pc1, pc2, pc2 (np.ndarray): Pointclouds to be visualized.
        title1, title2, title3 (str, optional): Titles for the diagrams. Defaults to "".
        show_axis (bool, optional): Show axis and background. Defaults to True.
        axis_ranges (Optional[List[Tuple[float, float]]], optional): Axis ranges. Defaults to None.
    """
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d', 'facecolor': 'none'}, figsize=(12, 6))

    for ax, pc, title in zip(axes, [pc1, pc2, pc3], [title1, title2, title3]):
        pc = rotate_point_cloud_y(pc, -90)
        pc = rotate_point_cloud_x(pc, 90)

        # Depth color mapping
        depth = pc[:, 2]
        depth_colormap = plt.get_cmap("viridis")

        # Plot
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=depth, cmap=depth_colormap)

        # Labels and title
        if show_axis:
            ax.set_xlabel("Z")
            ax.set_ylabel("X")
            ax.set_zlabel("Y")
            ax.set_title(title)

        # Setting the aspect ratio
        if axis_ranges is None:
            axis_ranges = get_axis_ranges(pc)

        ax.set_xlim(*axis_ranges[0])
        ax.set_ylim(*axis_ranges[1])
        ax.set_zlim(*axis_ranges[2])

        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.axis("off")

        ax.view_init(15, 0)

    plt.tight_layout()
    return fig