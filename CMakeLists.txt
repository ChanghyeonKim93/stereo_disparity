cmake_minimum_required(VERSION 2.8.3)
project(stereo_disparity)

find_package(catkin REQUIRED cmake_modules)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
  message_filters
)

catkin_package(
		DEPENDS Eigen
)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)

set(SRC ${SRC} 
	src/main_2.cpp
	src/StereoDisparity.cpp
	include/StereoDisparity.h
)

include_directories( ${includeDirs} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen_INCLUDE_DIRS} )

add_executable (stereo_disparity_node ${SRC} )
target_link_libraries (stereo_disparity_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries (stereo_disparity_node ${Eigen_LIBRARIES} )
