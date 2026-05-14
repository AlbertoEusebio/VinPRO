"""
VinPRO simulation-only launch.
Uses fake hardware controllers and a mock depth image publisher
so the full pipeline can be validated without physical hardware.

Usage:
    ros2 launch vinpro_bringup sim_only.launch.py \
        model_checkpoint:=/abs/path/to/model.pt

Test a single cutting point manually:
    ros2 topic pub /pixel_coordinates std_msgs/msg/Int32MultiArray \
        "data: [320, 240]" --once
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
        description="Path to ViNet weights (.pt) — leave empty to skip inference",
    )
    ld.add_action(ckpt_arg)

    # ── 1. Kinova with fake hardware (no physical arm required) ───────────────
    kinova_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("kinova_gen3_lite_moveit_config"),
            "launch", "robot.launch.py",
        )),
        launch_arguments={"use_fake_hardware": "true"}.items(),
    )
    ld.add_action(kinova_launch)

    # ── 2. Mock depth publisher (publishes a flat 640×480 depth frame at 1 m) ─
    mock_depth = Node(
        package="vinpro_bringup",
        executable="mock_depth_publisher",
        name="mock_depth_publisher",
        output="screen",
    )
    ld.add_action(mock_depth)

    # ── 3. Camera node ────────────────────────────────────────────────────────
    camera_node = Node(
        package="vinpro_camera",
        executable="camera_subscriber_node",
        name="vinpro_camera_node",
        output="screen",
    )
    ld.add_action(camera_node)

    # ── 4. Transform + MTC node ───────────────────────────────────────────────
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

    # ── 5. Static TF: camera_link → camera_color_optical_frame ───────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_tf",
        arguments=["0", "0", "0", "-1.5708", "0", "-1.5708",
                   "camera_link", "camera_color_optical_frame"],
    )
    ld.add_action(static_tf)

    return ld
