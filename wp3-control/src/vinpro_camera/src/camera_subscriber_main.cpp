#include <rclcpp/rclcpp.hpp>
#include "vinpro_camera/camera_subscriber_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vinpro_camera::CameraSubscriberNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
