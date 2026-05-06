#!/usr/bin/env python3
"""坐标转换节点 — 像素→相机系 3D→基座系，发布抓取目标位姿。

订阅 /detected_objects (PoseStamped, 编码像素信息)
发布 /grasp_target (PoseStamped, 基座坐标系)
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


class TfTransformerNode(Node):
    FX = 500.0
    FY = 500.0
    CX = 320.0
    CY = 240.0

    CAM_POS = np.array([0.15, 0.0, 0.5])
    CAM_R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])

    OBJECT_SIZES = {
        'red_block': 0.03,
        'green_block': 0.03,
        'blue_block': 0.025,
        'yellow_block': 0.028,
    }

    def __init__(self):
        super().__init__('tf_transformer_node')
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        self.sub_det = self.create_subscription(
            PoseStamped, '/detected_objects', self._det_cb, 10)
        self.pub_target = self.create_publisher(PoseStamped, '/grasp_target', 10)
        self.get_logger().info('坐标转换节点已启动')

    def _broadcast_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'Base'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.CAM_POS[0]
        t.transform.translation.y = self.CAM_POS[1]
        t.transform.translation.z = self.CAM_POS[2]
        t.transform.rotation.w = 0.0
        t.transform.rotation.x = 0.7071068
        t.transform.rotation.y = 0.7071068
        t.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info('已广播静态 TF: base_link → camera_link')

    def _pixel_to_cam3d(self, u, v, pixel_size, real_size):
        if pixel_size < 1.0:
            return None
        z_c = (real_size * self.FX) / pixel_size
        x_c = (u - self.CX) * z_c / self.FX
        y_c = (v - self.CY) * z_c / self.FY
        return np.array([x_c, y_c, z_c])

    def _cam_to_base(self, p_cam):
        return self.CAM_R @ p_cam + self.CAM_POS

    def _det_cb(self, msg: PoseStamped):
        obj_type = msg.header.frame_id  # 物体类型
        u = msg.pose.position.x
        v = msg.pose.position.y
        pixel_w = msg.pose.position.z

        real_size = self.OBJECT_SIZES.get(obj_type, 0.03)

        p_cam = self._pixel_to_cam3d(u, v, pixel_w, real_size)
        if p_cam is None:
            return

        p_base = self._cam_to_base(p_cam)

        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = 'Base'
        target.pose.position.x = float(p_base[0])
        target.pose.position.y = float(p_base[1])
        target.pose.position.z = float(max(p_base[2], 0.0) + 0.05)
        target.pose.orientation.w = 1.0
        self.pub_target.publish(target)

        self.get_logger().info(
            f'{obj_type}: 像素({u:.0f},{v:.0f}) → '
            f'基座({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = TfTransformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
