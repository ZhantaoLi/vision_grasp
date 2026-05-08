#!/usr/bin/env python3
"""坐标转换节点 — 像素→相机系 3D→基座系，发布抓取目标位姿。

订阅 /detected_objects (PoseStamped, 编码像素信息)
订阅 /camera_info (CameraInfo, 相机内参)
发布 /grasp_target (PoseStamped, 基座坐标系)
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tf2_ros import StaticTransformBroadcaster


# 相机旋转矩阵: Xc→+Xb, Yc→-Yb, Zc→-Zb (向下看)
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


class TfTransformerNode(Node):
    def __init__(self):
        super().__init__('tf_transformer_node')
        self.declare_parameter('target_interval', 10.0)
        self.declare_parameter('cam_pos_x', 0.15)
        self.declare_parameter('cam_pos_y', 0.0)
        self.declare_parameter('cam_pos_z', 0.5)

        self.target_interval = self.get_parameter('target_interval').value
        self.cam_pos = np.array([
            self.get_parameter('cam_pos_x').value,
            self.get_parameter('cam_pos_y').value,
            self.get_parameter('cam_pos_z').value,
        ])
        self._last_target_time = 0.0

        # 相机内参，从 /camera_info 更新
        self._fx = 500.0
        self._fy = 500.0
        self._cx = 320.0
        self._cy = 240.0

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        self.sub_info = self.create_subscription(
            CameraInfo, '/camera_info', self._info_cb, 10)
        self.sub_det = self.create_subscription(
            PoseStamped, '/detected_objects', self._det_cb, 10)
        self.pub_target = self.create_publisher(PoseStamped, '/grasp_target', 10)
        self.get_logger().info(
            f'坐标转换节点已启动 (目标间隔 {self.target_interval}s)')

    def _info_cb(self, msg: CameraInfo):
        k = msg.k
        self._fx = k[0]
        self._fy = k[4]
        self._cx = k[2]
        self._cy = k[5]

    def _broadcast_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.cam_pos[0]
        t.transform.translation.y = self.cam_pos[1]
        t.transform.translation.z = self.cam_pos[2]
        t.transform.rotation.w = 0.0
        t.transform.rotation.x = 1.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info('已广播静态 TF: base_link → camera_link')

    def _pixel_to_base(self, u, v):
        """像素 (u,v) → 基座系 z=0 平面上的 3D 点 (射线-平面求交)。"""
        d_cam = np.array([(u - self._cx) / self._fx,
                          (v - self._cy) / self._fy,
                          1.0])
        d_base = CAM_R @ d_cam
        if abs(d_base[2]) < 1e-9:
            return None
        t = -self.cam_pos[2] / d_base[2]
        if t < 0:
            return None
        return self.cam_pos + t * d_base

    def _det_cb(self, msg: PoseStamped):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_target_time < self.target_interval:
            return
        self._last_target_time = now

        obj_type = msg.header.frame_id
        u = msg.pose.position.x
        v = msg.pose.position.y

        p_base = self._pixel_to_base(u, v)
        if p_base is None:
            return

        block_w = OBJECT_SIZES.get(obj_type, 0.03)
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = 'base_link'
        target.pose.position.x = float(p_base[0])
        target.pose.position.y = float(p_base[1])
        target.pose.position.z = float(block_w)
        target.pose.orientation.w = 1.0
        target.pose.orientation.x = float(block_w)
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
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
