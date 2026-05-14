cmake_minimum_required(VERSION 3.8)
project(vinpro_mtc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_library(vinpro_mtc SHARED
  src/planning_task.cpp
)

target_include_directories(vinpro_mtc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vinpro_mtc
  rclcpp
  geometry_msgs
  moveit_task_constructor_core
  moveit_ros_planning_interface
)

install(TARGETS vinpro_mtc
  EXPORT vinpro_mtcTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_export_targets(vinpro_mtcTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp geometry_msgs moveit_task_constructor_core moveit_ros_planning_interface)

ament_package()
