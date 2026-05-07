#!/usr/bin/env python3
"""坐标转换节点 — 像素→相机系 3D→基座系，发布抓取目标位姿。

订阅 /detected_objects (PoseStamped, 编码像素信息)
发布 /grasp_target (PoseStamped, 基座坐标系)
"""

import math
import time

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
        'red_block': 0.035,
        'green_block': 0.035,
        'blue_block': 0.030,
        'yellow_block': 0.032,
    }

    def __init__(self):
        super().__init__('tf_transformer_node')
        self.declare_parameter('target_interval', 10.0)
        self.target_interval = self.get_parameter('target_interval').value
        self._last_target_time = 0.0

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        self.sub_det = self.create_subscription(
            PoseStamped, '/detected_objects', self._det_cb, 10)
        self.pub_target = self.create_publisher(PoseStamped, '/grasp_target', 10)
        self.get_logger().info(
            f'坐标转换节点已启动 (目标间隔 {self.target_interval}s)')

    def _broadcast_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.CAM_POS[0]
        t.transform.translation.y = self.CAM_POS[1]
        t.transform.translation.z = self.CAM_POS[2]
        t.transform.rotation.w = 0.0
        t.transform.rotation.x = 1.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info('已广播静态 TF: base_link → camera_link')

    def _pixel_to_base(self, u, v):
        """像素 (u,v) → 基座系 z=0 平面上的 3D 点 (射线-平面求交)。"""
        # 相机系中的射线方向
        d_cam = np.array([(u - self.CX) / self.FX,
                          (v - self.CY) / self.FY,
                          1.0])
        # 转到基座系
        d_base = self.CAM_R @ d_cam
        if abs(d_base[2]) < 1e-9:
            return None
        # 射线原点 + t*方向 与 z=0 平面求交
        t = -self.CAM_POS[2] / d_base[2]
        if t < 0:
            return None
        return self.CAM_POS + t * d_base

    def _det_cb(self, msg: PoseStamped):
        now = time.monotonic()
        if now - self._last_target_time < self.target_interval:
            return
        self._last_target_time = now

        obj_type = msg.header.frame_id
        u = msg.pose.position.x
        v = msg.pose.position.y

        p_base = self._pixel_to_base(u, v)
        if p_base is None:
            return

        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = 'base_link'
        target.pose.position.x = float(p_base[0])
        target.pose.position.y = float(p_base[1])
        target.pose.position.z = 0.05
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
