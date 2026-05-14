cmake_minimum_required(VERSION 3.8)
project(vinpro_transform)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(vinpro_mtc REQUIRED)

add_executable(transform_node
  src/transform_node.cpp
  src/transform_main.cpp
)

target_include_directories(transform_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(transform_node
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  visualization_msgs
  moveit_ros_planning_interface
  custom_msgs
  vinpro_mtc
)

install(TARGETS transform_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
