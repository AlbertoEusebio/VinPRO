#include "vinpro_transform/transform_node.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include "vinpro_mtc/planning_task.hpp"

namespace vinpro_transform
{

TransformNode::TransformNode(const rclcpp::NodeOptions & options)
: Node("vinpro_transform_node", options)
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  points_sub_ = this->create_subscription<custom_msgs::msg::PointsArray>(
    "/detected_goal_pose",
    10,
    [this](custom_msgs::msg::PointsArray::SharedPtr msg) {
      points_callback(msg);
    });

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/cutting_point_markers", 10);

  RCLCPP_INFO(get_logger(), "TransformNode ready — waiting for cutting points");
}

void TransformNode::points_callback(
  const custom_msgs::msg::PointsArray::SharedPtr msg)
{
  const std::string source_frame = msg->header.frame_id;
  const std::string target_frame = "base_link";

  geometry_msgs::msg::TransformStamped tf_stamped;
  try {
    tf_stamped = tf_buffer_->lookupTransform(
      target_frame, source_frame, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(get_logger(),
      "TF lookup '%s' → '%s' failed: %s",
      source_frame.c_str(), target_frame.c_str(), ex.what());
    return;
  }

  std::vector<geometry_msgs::msg::Point> global_points;
  visualization_msgs::msg::MarkerArray markers;

  for (size_t i = 0; i < msg->points.size(); ++i) {
    geometry_msgs::msg::PointStamped cam_pt, base_pt;
    cam_pt.header = msg->header;
    cam_pt.point = msg->points[i];

    tf2::doTransform(cam_pt, base_pt, tf_stamped);
    global_points.push_back(base_pt.point);

    // RViz sphere marker at each cutting point
    visualization_msgs::msg::Marker m;
    m.header.frame_id = target_frame;
    m.header.stamp = this->now();
    m.ns = "cutting_points";
    m.id = static_cast<int>(i);
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position = base_pt.point;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.02;  // 2 cm
    m.color.r = 1.0f;
    m.color.a = 1.0f;
    m.lifetime = rclcpp::Duration::from_seconds(10.0);
    markers.markers.push_back(m);
  }

  marker_pub_->publish(markers);
  RCLCPP_INFO(get_logger(),
    "Transformed %zu cutting point(s) to '%s'",
    global_points.size(), target_frame.c_str());

  if (global_points.empty()) {
    return;
  }

  // ── Build and execute MTC plan ───────────────────────────────────────────
  vinpro_mtc::PlanningTask task(shared_from_this());
  task.init(global_points);

  bool success = false;
  constexpr int kMaxAttempts = 10;
  for (int attempt = 0; attempt < kMaxAttempts && !success; ++attempt) {
    success = task.plan();
    if (!success) {
      RCLCPP_WARN(get_logger(),
        "MTC planning attempt %d/%d failed", attempt + 1, kMaxAttempts);
    }
  }

  if (success) {
    RCLCPP_INFO(get_logger(), "MTC plan found — executing");
    task.execute();
    RCLCPP_INFO(get_logger(), "Execution complete");
  } else {
    RCLCPP_ERROR(get_logger(),
      "MTC planning failed after %d attempts — aborting cut sequence",
      kMaxAttempts);
  }
}

}  // namespace vinpro_transform
