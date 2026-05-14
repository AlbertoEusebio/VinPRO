cmake_minimum_required(VERSION 3.8)
project(vinpro_arduino)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)

add_library(vinpro_arduino SHARED
  src/scissors_controller.cpp
)

target_include_directories(vinpro_arduino PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vinpro_arduino
  rclcpp
  rclcpp_lifecycle
  hardware_interface
  pluginlib
)

pluginlib_export_plugin_description_file(
  hardware_interface vinpro_arduino_plugin.xml)

install(TARGETS vinpro_arduino
  EXPORT vinpro_arduinoTargets
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

install(FILES vinpro_arduino_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(vinpro_arduinoTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp rclcpp_lifecycle hardware_interface pluginlib)
ament_package()
