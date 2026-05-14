cmake_minimum_required(VERSION 3.8)
project(vinpro_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(custom_msgs REQUIRED)

add_executable(camera_subscriber_node
  src/camera_subscriber_node.cpp
  src/camera_subscriber_main.cpp
)

target_include_directories(camera_subscriber_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(camera_subscriber_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  image_transport
  cv_bridge
  OpenCV
  custom_msgs
)

install(TARGETS camera_subscriber_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
