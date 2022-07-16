cmake_minimum_required(VERSION 3.10)
project(face_detection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_definitions(-Wall -Werror)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  nodelet
  message_generation
)

catkin_package(
  INCLUDE_DIRS
  include
#  LIBRARIES face_detection
  CATKIN_DEPENDS
  roscpp
  rospy
  roslib
  std_msgs
  sensor_msgs
  image_transport
  nodelet
  DEPENDS
  system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/${PROJECT_NAME}.cpp
)

target_link_libraries(face_detection
  ${catkin_LIBRARIES}
)