#!/usr/bin/env python3
"""摄像头节点 — 模拟模式生成测试图像，或使用真实摄像头。"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('use_camera', False)
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.bridge = CvBridge()

        # 相机内参
        self.fx = 500.0
        self.fy = 500.0
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # 相机在基座坐标系中的位姿 (向下看)
        self.cam_pos = np.array([0.15, 0.0, 0.5])
        # 旋转矩阵: Xc→+Xb, Yc→-Yb, Zc→-Zb
        self.cam_R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

        # 仿真场景中的方块 (frame, x_m, y_m, size_m, bgr_color)
        self.blocks = [
            ('red_block', 0.12, 0.06, 0.03, (0, 0, 200)),
            ('green_block', 0.06, -0.08, 0.03, (0, 180, 0)),
            ('blue_block', 0.18, -0.04, 0.025, (200, 100, 0)),
            ('yellow_block', 0.10, 0.10, 0.028, (0, 200, 220)),
        ]

        self.pub_image = self.create_publisher(Image, '/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera_info', 10)

        use_camera = self.get_parameter('use_camera').value
        if use_camera:
            cam_id = self.get_parameter('camera_id').value
            self.cap = cv2.VideoCapture(cam_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.get_logger().info(f'真实摄像头模式 (id={cam_id})')
        else:
            self.cap = None
            self.get_logger().info('仿真模式 — 生成测试图像')

        fps = self.get_parameter('fps').value
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

    def _project(self, xyz_base):
        """将基座系 3D 点投影到像素坐标 (u, v)。"""
        p_cam = self.cam_R.T @ (xyz_base - self.cam_pos)
        x_c, y_c, z_c = p_cam
        if abs(z_c) < 1e-9:
            return None
        u = self.fx * x_c / z_c + self.cx
        v = self.fy * y_c / z_c + self.cy
        return int(round(u)), int(round(v))

    def _draw_blocks(self, img):
        """在图像上绘制仿真方块。"""
        for _name, bx, by, bsize, color in self.blocks:
            half = bsize / 2.0
            corners_base = np.array([
                [bx - half, by - half, 0.0],
                [bx + half, by - half, 0.0],
                [bx + half, by + half, 0.0],
                [bx - half, by + half, 0.0],
            ])
            pts = []
            for c in corners_base:
                uv = self._project(c)
                if uv is None:
                    break
                pts.append(uv)
            if len(pts) == 4:
                cv2.fillPoly(img, [np.array(pts)], color)
                cv2.polylines(img, [np.array(pts)], True, (0, 0, 0), 2)

    def publish_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('摄像头读取失败')
                return
        else:
            frame = np.full((self.height, self.width, 3), 50, dtype=np.uint8)
            self._draw_blocks(frame)

        # 发布图像
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        self.pub_image.publish(msg)

        # 发布 CameraInfo
        info = CameraInfo()
        info.header = msg.header
        info.width = self.width
        info.height = self.height
        info.k = [self.fx, 0.0, self.cx,
                  0.0, self.fy, self.cy,
                  0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, self.cx, 0.0,
                  0.0, self.fy, self.cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_info.publish(info)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
