import numpy as np


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
