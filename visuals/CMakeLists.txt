cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pcl_mls_example)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(mls_example main.cpp)
target_link_libraries(mls_example ${PCL_LIBRARIES} pcl_surface)
