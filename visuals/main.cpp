#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>

int main()
{
    // Load the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPLYFile("/Users/stephanschmiedmayer/Desktop/pc_10k.ply", *cloud);
    if (cloud->empty())
    {
        std::cerr << "Failed to load point cloud." << std::endl;
        return -1;
    }

    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // Initialize MLS object
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZINormal> mls;
    mls.setInputCloud(cloud);
    mls.setSearchMethod(tree);
    mls.setComputeNormals(true);

    // Set parameters for MLS
    mls.setSearchRadius(0.01); // Example value, adjust based on your data

    // Output has the PointXYZINormal type in order to store the points and normals
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr mls_points(new pcl::PointCloud<pcl::PointXYZINormal>);
    mls.process(*mls_points);

    if (mls_points->empty())
    {
        std::cerr << "MLS processing resulted in an empty point cloud." << std::endl;
        return -1;
    }

    // Save to file
    pcl::io::savePLYFile("/Users/stephanschmiedmayer/Desktop/normals_pcl_10k.ply", *mls_points);

    return 0;
}
