import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo

from vision_grasp.tf_transformer_node import TfTransformerNode


def setup_module():
    rclpy.init()


def teardown_module():
    rclpy.shutdown()


def test_pixel_to_base_projects_center_pixel_to_table_point():
    node = TfTransformerNode()
    try:
        point = node._pixel_to_base(320.0, 240.0)
        assert np.allclose(point, np.array([0.15, 0.0, 0.0]), atol=1e-6)
    finally:
        node.destroy_node()


def test_camera_info_callback_updates_intrinsics():
    node = TfTransformerNode()
    try:
        msg = CameraInfo()
        msg.k = [600.0, 0.0, 300.0, 0.0, 610.0, 200.0, 0.0, 0.0, 1.0]
        node._info_cb(msg)

        assert node._fx == 600.0
        assert node._fy == 610.0
        assert node._cx == 300.0
        assert node._cy == 200.0
    finally:
        node.destroy_node()
