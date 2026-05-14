"""
VinPRO full system launch.
Starts: Kinova driver + MoveIt, RealSense, WP2 inference, camera node,
        transform+MTC node, and the Arduino hardware interface.

Usage:
    ros2 launch vinpro_bringup full_system.launch.py \
        model_checkpoint:=/abs/path/to/model.pt
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    ckpt_arg = DeclareLaunchArgument(
        "model_checkpoint",
        default_value="",
        description="Absolute path to the trained ViNet weights (.pt)",
    )
    ld.add_action(ckpt_arg)

    # ── 1. Kinova Gen3 Lite driver + MoveIt 2 ────────────────────────────────
    # Requires ros2_kortex and kinova_gen3_lite_moveit_config to be installed.
    kinova_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("kinova_gen3_lite_moveit_config"),
            "launch", "robot.launch.py",
        )),
    )
    ld.add_action(kinova_launch)

    # ── 2. Intel RealSense D435i ──────────────────────────────────────────────
    # Requires realsense2_camera ROS 2 package.
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("realsense2_camera"),
            "launch", "rs_launch.py",
        )),
        launch_arguments={
            "align_depth.enable": "true",
            "pointcloud.enable": "false",
        }.items(),
    )
    ld.add_action(realsense_launch)

    # ── 3. WP2 perception + pruning policy (Python) ───────────────────────────
    vinpro_perception_pkg = get_package_share_directory("vinpro_perception")
    perception_node = Node(
        package="vinpro_perception",
        executable="inference_node",
        name="vinpro_inference",
        parameters=[
            os.path.join(vinpro_perception_pkg, "config", "perception_params.yaml"),
            {"model_checkpoint": LaunchConfiguration("model_checkpoint")},
        ],
        output="screen",
    )
    ld.add_action(perception_node)

    # ── 4. Camera depth → 3-D node (C++) ──────────────────────────────────────
    camera_node = Node(
        package="vinpro_camera",
        executable="camera_subscriber_node",
        name="vinpro_camera_node",
        output="screen",
    )
    ld.add_action(camera_node)

    # ── 5. TF2 transform + MTC launcher (C++) ─────────────────────────────────
    mtc_params = os.path.join(
        get_package_share_directory("vinpro_mtc"), "config", "mtc_params.yaml"
    )
    transform_node = Node(
        package="vinpro_transform",
        executable="transform_node",
        name="vinpro_transform_node",
        parameters=[mtc_params],
        output="screen",
    )
    ld.add_action(transform_node)

    return ld
