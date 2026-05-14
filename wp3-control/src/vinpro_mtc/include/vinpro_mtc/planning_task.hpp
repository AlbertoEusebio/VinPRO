#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <moveit/task_constructor/task.h>

namespace vinpro_mtc
{

/**
 * @brief Builds and executes a MoveIt Task Constructor pipeline for
 *        vine pruning: for each cutting point the arm approaches,
 *        closes the shears, opens them, and retreats.
 *
 * Parameters (set via mtc_params.yaml):
 *   arm_group       — MoveIt planning group for the arm  (default: "manipulator")
 *   scissors_group  — Single-joint group for the shear   (default: "scissors")
 *   eef_frame       — End-effector frame for MTC         (default: "scissors_tip")
 *   scissors_joint  — ros2_control joint name            (default: "scissors_joint")
 *   approach_dist   — Linear approach distance [m]       (default: 0.05)
 *   retreat_dist    — Linear retreat distance [m]        (default: 0.05)
 */
class PlanningTask
{
public:
  explicit PlanningTask(rclcpp::Node::SharedPtr node);

  /** Initialise the task for the given list of 3-D cutting targets (base_link). */
  void init(const std::vector<geometry_msgs::msg::Point> & targets);

  /** Run the planner. Returns true when a solution is found. */
  bool plan();

  /** Execute the best solution found by the last plan() call. */
  void execute();

private:
  void load_parameters();
  void build_pipeline(const std::vector<geometry_msgs::msg::Point> & targets);

  rclcpp::Node::SharedPtr node_;
  moveit::task_constructor::TaskPtr task_;

  // Config
  std::string arm_group_;
  std::string scissors_group_;
  std::string eef_frame_;
  std::string scissors_joint_;
  double approach_dist_;
  double retreat_dist_;
};

}  // namespace vinpro_mtc
