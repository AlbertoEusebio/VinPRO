#include "vinpro_mtc/planning_task.hpp"

#include <map>
#include <stdexcept>

#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/connect.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace vinpro_mtc
{

namespace mtc = moveit::task_constructor;

PlanningTask::PlanningTask(rclcpp::Node::SharedPtr node)
: node_(node)
{
  load_parameters();
}

void PlanningTask::load_parameters()
{
  node_->declare_parameter("arm_group",      "manipulator");
  node_->declare_parameter("scissors_group", "scissors");
  node_->declare_parameter("eef_frame",      "scissors_tip");
  node_->declare_parameter("scissors_joint", "scissors_joint");
  node_->declare_parameter("approach_dist",   0.05);
  node_->declare_parameter("retreat_dist",    0.05);

  arm_group_      = node_->get_parameter("arm_group").as_string();
  scissors_group_ = node_->get_parameter("scissors_group").as_string();
  eef_frame_      = node_->get_parameter("eef_frame").as_string();
  scissors_joint_ = node_->get_parameter("scissors_joint").as_string();
  approach_dist_  = node_->get_parameter("approach_dist").as_double();
  retreat_dist_   = node_->get_parameter("retreat_dist").as_double();
}

void PlanningTask::init(
  const std::vector<geometry_msgs::msg::Point> & targets)
{
  task_ = std::make_shared<mtc::Task>();
  task_->loadRobotModel(node_);
  build_pipeline(targets);
}

void PlanningTask::build_pipeline(
  const std::vector<geometry_msgs::msg::Point> & targets)
{
  // ── Shared solvers ────────────────────────────────────────────────────────
  auto cartesian = std::make_shared<mtc::solvers::CartesianPath>();
  cartesian->setMaxVelocityScalingFactor(0.3);
  cartesian->setMaxAccelerationScalingFactor(0.3);
  cartesian->setStepSize(0.005);

  auto pipeline = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
  pipeline->setPlannerId("RRTConnect");

  // ── Stage 0: latch current robot state ───────────────────────────────────
  task_->add(std::make_unique<mtc::stages::CurrentState>("current_state"));

  // ── Per-target stages ─────────────────────────────────────────────────────
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto & pt = targets[i];
    const std::string lbl = "cut_" + std::to_string(i);

    // Free-space move to a configuration near the cutting point
    {
      mtc::stages::Connect::GroupPlannerVector planners{
        {arm_group_, pipeline}};
      task_->add(std::make_unique<mtc::stages::Connect>(
        lbl + "_connect", planners));
    }

    // Cartesian approach along the tool z-axis
    {
      auto approach = std::make_unique<mtc::stages::MoveRelative>(
        lbl + "_approach", cartesian);
      approach->setGroup(arm_group_);
      geometry_msgs::msg::Vector3Stamped dir;
      dir.header.frame_id = eef_frame_;
      dir.vector.z = approach_dist_;
      approach->setDirection(dir);
      task_->add(std::move(approach));
    }

    // Move end-effector to the exact cutting point
    {
      auto move_to = std::make_unique<mtc::stages::MoveTo>(
        lbl + "_move_to", pipeline);
      move_to->setGroup(arm_group_);

      geometry_msgs::msg::PoseStamped target_pose;
      target_pose.header.frame_id = "base_link";
      target_pose.pose.position.x = pt.x;
      target_pose.pose.position.y = pt.y;
      target_pose.pose.position.z = pt.z;
      // TODO: when WP2 branch-direction orientation is available, set it here.
      // For now use identity (unoriented cut).
      target_pose.pose.orientation.w = 1.0;

      move_to->setGoal(target_pose);
      task_->add(std::move(move_to));
    }

    // Close scissors (scissors_joint → 1.0 = closed)
    {
      auto cut = std::make_unique<mtc::stages::MoveTo>(
        lbl + "_cut", pipeline);
      cut->setGroup(scissors_group_);
      std::map<std::string, double> closed{{scissors_joint_, 1.0}};
      cut->setGoal(closed);
      task_->add(std::move(cut));
    }

    // Open scissors (scissors_joint → 0.0 = open)
    {
      auto open = std::make_unique<mtc::stages::MoveTo>(
        lbl + "_open", pipeline);
      open->setGroup(scissors_group_);
      std::map<std::string, double> opened{{scissors_joint_, 0.0}};
      open->setGoal(opened);
      task_->add(std::move(open));
    }

    // Cartesian retreat along the tool z-axis
    {
      auto retreat = std::make_unique<mtc::stages::MoveRelative>(
        lbl + "_retreat", cartesian);
      retreat->setGroup(arm_group_);
      geometry_msgs::msg::Vector3Stamped dir;
      dir.header.frame_id = eef_frame_;
      dir.vector.z = -retreat_dist_;
      retreat->setDirection(dir);
      task_->add(std::move(retreat));
    }
  }

  // ── Return to home ────────────────────────────────────────────────────────
  {
    auto home = std::make_unique<mtc::stages::MoveTo>("return_home", pipeline);
    home->setGroup(arm_group_);
    home->setGoal("home");  // named state defined in the SRDF
    task_->add(std::move(home));
  }
}

bool PlanningTask::plan()
{
  try {
    auto error_code = task_->plan(10 /* max solutions */);
    return error_code == moveit::core::MoveItErrorCode::SUCCESS;
  } catch (const moveit::task_constructor::InitStageException & e) {
    RCLCPP_ERROR(node_->get_logger(), "MTC init stage failed: %s", e.what());
    return false;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node_->get_logger(), "MTC planning exception: %s", e.what());
    return false;
  }
}

void PlanningTask::execute()
{
  if (task_->solutions().empty()) {
    RCLCPP_ERROR(node_->get_logger(),
      "execute() called but no solutions available");
    return;
  }
  task_->execute(*task_->solutions().front());
}

}  // namespace vinpro_mtc
