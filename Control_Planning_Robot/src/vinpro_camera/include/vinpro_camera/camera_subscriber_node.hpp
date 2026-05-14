#pragma once

#include <mutex>
#include <vector>
#include <utility>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "custom_msgs/msg/points_array.hpp"

namespace vinpro_camera
{

/**
 * @brief Converts (pixel, depth) pairs from WP2 into 3-D camera-frame points.
 *
 * Subscribes to:
 *   /pixel_coordinates                         std_msgs/Int32MultiArray
 *   /camera/aligned_depth_to_color/image_raw   sensor_msgs/Image
 *   /camera/aligned_depth_to_color/camera_info sensor_msgs/CameraInfo
 *
 * Publishes to:
 *   /detected_goal_pose                        custom_msgs/PointsArray
 */
class CameraSubscriberNode : public rclcpp::Node
{
public:
  explicit CameraSubscriberNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions{});

private:
  // Callbacks
  void depth_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg);
  void pixel_callback(const std_msgs::msg::Int32MultiArray::SharedPtr msg);

  // Pinhole back-projection: (u, v, depth_m) → (x, y, z) in camera frame
  static void deproject_pixel(
    int u, int v, float depth_m,
    const sensor_msgs::msg::CameraInfo & info,
    float & x, float & y, float & z);

  // Latest depth frame and camera intrinsics (guarded by depth_mutex_)
  cv::Mat latest_depth_;
  sensor_msgs::msg::CameraInfo latest_info_;
  std::mutex depth_mutex_;

  // Buffered pixel coordinates from WP2 (guarded by pixel_mutex_)
  std::vector<std::pair<int, int>> pixels_;
  std::mutex pixel_mutex_;

  // ROS interfaces
  image_transport::CameraSubscriber depth_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr pixel_sub_;
  rclcpp::Publisher<custom_msgs::msg::PointsArray>::SharedPtr points_pub_;
};

}  // namespace vinpro_camera
