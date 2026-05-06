"""启动视觉抓取管线 (不含 RViz 和 Joint State GUI)。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_grasp')
    urdf_path = os.path.join(pkg_share, 'description', 'genkiarm.urdf')
    config_path = os.path.join(pkg_share, 'config', 'params.yaml')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        LogInfo(msg=['=== 视觉抓取管线启动 ===']),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),

        Node(
            package='vision_grasp',
            executable='camera_node',
            name='camera_node',
            parameters=[config_path],
        ),

        Node(
            package='vision_grasp',
            executable='vision_node',
            name='vision_node',
            parameters=[config_path],
        ),

        Node(
            package='vision_grasp',
            executable='tf_transformer_node',
            name='tf_transformer_node',
        ),

        Node(
            package='vision_grasp',
            executable='ik_solver_node',
            name='ik_solver_node',
            parameters=[config_path],
        ),

        Node(
            package='vision_grasp',
            executable='arm_driver_node',
            name='arm_driver_node',
            parameters=[config_path],
        ),

        LogInfo(msg=['=== 管线已启动 (无 RViz) ===']),
    ])
