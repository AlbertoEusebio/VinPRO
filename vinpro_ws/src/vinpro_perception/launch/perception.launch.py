from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("vinpro_perception")

    ckpt_arg = DeclareLaunchArgument(
        "model_checkpoint",
        default_value="",
        description="Path to trained ViNet weights (.pt)",
    )

    inference_node = Node(
        package="vinpro_perception",
        executable="inference_node",
        name="vinpro_inference",
        parameters=[
            os.path.join(pkg, "config", "perception_params.yaml"),
            {"model_checkpoint": LaunchConfiguration("model_checkpoint")},
        ],
        output="screen",
    )

    return LaunchDescription([ckpt_arg, inference_node])
