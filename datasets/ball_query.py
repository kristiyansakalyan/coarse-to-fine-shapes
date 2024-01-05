import numpy as np


def ball_query(point_cloud, centroid, radius):
    """
    Perform a ball query on a point cloud.

    Args:
    - point_cloud (np.ndarray): The point cloud, assumed to be an Nx3 array.
    - centroid (np.ndarray): The centroid point, a 1x3 array.
    - radius (float): The radius of the sphere within which to search for points.

    Returns:
    - np.ndarray: An array of points within the specified radius of the centroid.
    """
    # Calculate squared distances from the centroid to each point
    distances_squared = np.sum((point_cloud - centroid) ** 2, axis=1)

    # Find points where the squared distance is less than the squared radius
    within_radius = distances_squared < radius**2

    # Return points within the radius
    return point_cloud[within_radius]
