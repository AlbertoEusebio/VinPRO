"""
VinPRO hardware launch.
Identical to full_system.launch.py but loads the scissors hardware interface
and activates the Arduino serial connection.

Requires:
  - Kinova Gen3 Lite connected via USB
  - Intel RealSense D435i connected via USB
  - Arduino Nano connected via USB (check /dev/ttyACM0 or /dev/ttyUSB0)

Usage:
    ros2 launch vinpro_bringup hardware.launch.py \
        model_checkpoint:=/abs/path/to/model.pt \
        serial_port:=/dev/ttyACM0
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
        "model_checkpoint", default_value="",
        description="Absolute path to trained ViNet weights (.pt)",
    )
    serial_arg = DeclareLaunchArgument(
        "serial_port", default_value="/dev/ttyACM0",
        description="Arduino Nano serial device",
    )
    ld.add_action(ckpt_arg)
    ld.add_action(serial_arg)

    # ── Full system (reuse full_system.launch.py) ─────────────────────────────
    full = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("vinpro_bringup"),
            "launch", "full_system.launch.py",
        )),
        launch_arguments={
            "model_checkpoint": LaunchConfiguration("model_checkpoint"),
        }.items(),
    )
    ld.add_action(full)

    # ── Arduino hardware interface (loaded as a separate controller manager) ──
    scissors_hw = Node(
        package="controller_manager",
        executable="ros2_control_node",
        name="scissors_controller_manager",
        parameters=[
            os.path.join(
                get_package_share_directory("vinpro_arduino"),
                "config", "scissors_hw.yaml",
            ),
            {"serial_port": LaunchConfiguration("serial_port")},
        ],
        output="screen",
    )
    ld.add_action(scissors_hw)

    return ld
