from pathlib import Path
import time
import unittest

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "pipeline.launch.py"
EXPECTED_NODES = {
    "robot_state_publisher",
    "camera_node",
    "vision_node",
    "tf_transformer_node",
    "trajectory_node",
    "arm_driver_node",
}


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(LAUNCH_FILE))),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPipelineLaunch(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = rclpy.create_node("vision_grasp_pipeline_launch_test")

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_expected_nodes_start(self):
        deadline = time.time() + 20.0
        while time.time() < deadline:
            running_nodes = set(self.node.get_node_names())
            if EXPECTED_NODES.issubset(running_nodes):
                return
            time.sleep(0.1)

        running_nodes = set(self.node.get_node_names())
        missing = sorted(EXPECTED_NODES - running_nodes)
        assert not missing, f"Expected nodes did not appear: {missing}"
