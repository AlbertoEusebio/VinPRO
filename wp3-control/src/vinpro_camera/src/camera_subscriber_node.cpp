#include "vinpro_camera/camera_subscriber_node.hpp"
#include <geometry_msgs/msg/point.hpp>

namespace vinpro_camera
{

CameraSubscriberNode::CameraSubscriberNode(const rclcpp::NodeOptions & options)
: Node("vinpro_camera_node", options)
{
  // ── Depth + CameraInfo via image_transport ──────────────────────────────
  auto it = std::make_shared<image_transport::ImageTransport>(shared_from_this());
  depth_sub_ = it->subscribeCamera(
    "/camera/aligned_depth_to_color/image_raw",
    10,
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & d,
           const sensor_msgs::msg::CameraInfo::ConstSharedPtr & i) {
      depth_callback(d, i);
    });

  // ── Pixel coordinates from WP2 inference node ───────────────────────────
  pixel_sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    "/pixel_coordinates",
    10,
    [this](std_msgs::msg::Int32MultiArray::SharedPtr msg) {
      pixel_callback(msg);
    });

  // ── 3-D points in camera frame ──────────────────────────────────────────
  points_pub_ = this->create_publisher<custom_msgs::msg::PointsArray>(
    "/detected_goal_pose", 10);

  RCLCPP_INFO(get_logger(), "CameraSubscriberNode ready");
}

void CameraSubscriberNode::depth_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    // D435i aligned depth is 16-bit unsigned, millimetres
    cv_ptr = cv_bridge::toCvCopy(depth_msg, "16UC1");
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge depth conversion failed: %s", e.what());
    return;
  }

  std::lock_guard<std::mutex> lock(depth_mutex_);
  latest_depth_ = cv_ptr->image;
  latest_info_ = *info_msg;
}

void CameraSubscriberNode::pixel_callback(
  const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
  // Parse flat [u1, v1, u2, v2, ...] array
  if (msg->data.size() % 2 != 0) {
    RCLCPP_WARN(get_logger(), "Received odd-length pixel array — ignoring");
    return;
  }

  std::lock_guard<std::mutex> dlock(depth_mutex_);
  if (latest_depth_.empty()) {
    RCLCPP_WARN(get_logger(), "No depth frame received yet — dropping pixel message");
    return;
  }

  custom_msgs::msg::PointsArray out;
  out.header.stamp = this->now();
  out.header.frame_id = latest_info_.header.frame_id;  // camera_color_optical_frame

  for (size_t i = 0; i + 1 < msg->data.size(); i += 2) {
    int u = msg->data[i];
    int v = msg->data[i + 1];

    if (u < 0 || v < 0 || u >= latest_depth_.cols || v >= latest_depth_.rows) {
      RCLCPP_WARN(get_logger(),
        "Pixel (%d, %d) is outside depth image (%d×%d) — skipping",
        u, v, latest_depth_.cols, latest_depth_.rows);
      continue;
    }

    // Depth in mm → metres
    uint16_t depth_mm = latest_depth_.at<uint16_t>(v, u);
    if (depth_mm == 0) {
      RCLCPP_WARN(get_logger(), "Zero depth at (%d, %d) — skipping", u, v);
      continue;
    }
    float depth_m = static_cast<float>(depth_mm) / 1000.0f;

    float x, y, z;
    deproject_pixel(u, v, depth_m, latest_info_, x, y, z);

    geometry_msgs::msg::Point pt;
    pt.x = static_cast<double>(x);
    pt.y = static_cast<double>(y);
    pt.z = static_cast<double>(z);
    out.points.push_back(pt);

    // Pad orientations with identity so PointsArray stays consistent.
    // The transform node overwrites these when branch-direction data arrives.
    geometry_msgs::msg::Quaternion q;
    q.w = 1.0;
    out.orientations.push_back(q);
  }

  if (!out.points.empty()) {
    points_pub_->publish(out);
    RCLCPP_INFO(get_logger(),
      "Published %zu 3-D cutting point(s) in frame '%s'",
      out.points.size(), out.header.frame_id.c_str());
  }
}

void CameraSubscriberNode::deproject_pixel(
  int u, int v, float depth_m,
  const sensor_msgs::msg::CameraInfo & info,
  float & x, float & y, float & z)
{
  // Pinhole model using K = [fx  0  cx; 0  fy  cy; 0  0  1]
  // info.k is row-major: k[0]=fx, k[2]=cx, k[4]=fy, k[5]=cy
  float fx = static_cast<float>(info.k[0]);
  float fy = static_cast<float>(info.k[4]);
  float cx = static_cast<float>(info.k[2]);
  float cy = static_cast<float>(info.k[5]);

  // For D435i aligned depth the distortion is already removed; skip D matrix.
  x = (static_cast<float>(u) - cx) * depth_m / fx;
  y = (static_cast<float>(v) - cy) * depth_m / fy;
  z = depth_m;
}

}  // namespace vinpro_camera
