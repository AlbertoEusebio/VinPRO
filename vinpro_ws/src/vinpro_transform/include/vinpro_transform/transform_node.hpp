#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "custom_msgs/msg/points_array.hpp"

namespace vinpro_transform
{

/**
 * @brief Transforms 3-D cutting points from camera frame to base_link,
 *        visualises them in RViz, and triggers MTC planning + execution.
 *
 * Subscribes to:  /detected_goal_pose   (custom_msgs/PointsArray)
 * Publishes to:   /cutting_point_markers (visualization_msgs/MarkerArray)
 */
class TransformNode : public rclcpp::Node
{
public:
  explicit TransformNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

private:
  void points_callback(const custom_msgs::msg::PointsArray::SharedPtr msg);

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<custom_msgs::msg::PointsArray>::SharedPtr points_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

}  // namespace vinpro_transform
