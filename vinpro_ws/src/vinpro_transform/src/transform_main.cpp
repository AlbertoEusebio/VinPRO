#include <rclcpp/rclcpp.hpp>
#include "vinpro_transform/transform_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vinpro_transform::TransformNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
