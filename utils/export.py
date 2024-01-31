def add_extension_if_missing(filename, extension):
    """
    Adds an extension to a filename if it is missing.

    :param filename: Name of the file.
    :param extension: Extension to add to the filename.
    :return: Filename with the extension added.
    """
    if not filename.endswith(extension):
        filename += extension
    return filename


def export_pointcloud_to_pts(tensor, filename):
    """
    Exports a PyTorch point cloud tensor to a PTS file.

    :param tensor: PyTorch tensor of shape (N, 3) representing the point cloud.
    :param filename: Name of the file to save the point cloud to.
    """
    filename = add_extension_if_missing(filename, ".pts")
    # Ensure the tensor is on CPU and convert to numpy
    numpy_pc = tensor.cpu().numpy()

    # Open file and write points
    with open(filename, "w", encoding="utf-8") as file:
        for point in numpy_pc:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Point cloud exported to {filename}")


def export_pointcloud_to_asc(tensor, filename):
    """
    Exports a PyTorch point cloud tensor to an ASC file.

    :param tensor: PyTorch tensor representing the point cloud. Shape should be (N, 3) or (N, 6).
    :param filename: Name of the file to save the point cloud to.
    """
    filename = add_extension_if_missing(filename, ".asc")
    # Ensure the tensor is on CPU and convert to numpy
    numpy_pc = tensor.cpu().numpy()

    # Open file and write points
    with open(filename, "w", encoding="utf-8") as file:
        for point in numpy_pc:
            line = " ".join(map(str, point))
            file.write(f"{line}\n")

    print(f"Point cloud exported to {filename}")
