#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>

int main()
{
    // Load the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPLYFile("/Users/stephanschmiedmayer/Desktop/pc_10k.ply", *cloud);

    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    // Initialize MLS object
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setInputCloud(cloud);
    mls.setComputeNormals(true);

    // Set parameters for MLS
    // ...

    // Output has the PointNormal type in order to store the normals calculated by MLS
    pcl::PointCloud<pcl::PointNormal> mls_points;
    mls.process(mls_points);

    // Save to file
    pcl::io::savePLYFile("output_with_normals.ply", mls_points);

    return 0;
}
