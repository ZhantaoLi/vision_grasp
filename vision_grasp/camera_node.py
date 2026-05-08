#!/usr/bin/env python3
"""摄像头节点 — 模拟模式生成测试图像，或使用真实摄像头。"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_SIM_BLOCKS = [
    ('red_block', 0.30, 0.15, 0.035, (0, 0, 200)),
    ('green_block', 0.00, -0.18, 0.035, (0, 180, 0)),
    ('blue_block', 0.35, -0.12, 0.030, (200, 100, 0)),
    ('yellow_block', 0.05, 0.20, 0.032, (0, 200, 220)),
]


def build_sim_blocks(
    block_names,
    block_xs,
    block_ys,
    block_sizes,
    block_color_bs,
    block_color_gs,
    block_color_rs,
):
    lengths = [
        len(block_names),
        len(block_xs),
        len(block_ys),
        len(block_sizes),
        len(block_color_bs),
        len(block_color_gs),
        len(block_color_rs),
    ]
    if len(set(lengths)) != 1:
        raise ValueError('All block parameter arrays must have the same length')

    blocks = []
    for name, bx, by, bsize, blue, green, red in zip(
        block_names,
        block_xs,
        block_ys,
        block_sizes,
        block_color_bs,
        block_color_gs,
        block_color_rs,
    ):
        blocks.append(
            (str(name), float(bx), float(by), float(bsize), (int(blue), int(green), int(red)))
        )
    return blocks


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('use_camera', False)
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('cam_pos_x', 0.15)
        self.declare_parameter('cam_pos_y', 0.0)
        self.declare_parameter('cam_pos_z', 0.5)
        self.declare_parameter(
            'block_names',
            [name for name, _bx, _by, _size, _color in DEFAULT_SIM_BLOCKS],
        )
        self.declare_parameter(
            'block_xs',
            [bx for _name, bx, _by, _size, _color in DEFAULT_SIM_BLOCKS],
        )
        self.declare_parameter(
            'block_ys',
            [by for _name, _bx, by, _size, _color in DEFAULT_SIM_BLOCKS],
        )
        self.declare_parameter(
            'block_sizes',
            [bsize for _name, _bx, _by, bsize, _color in DEFAULT_SIM_BLOCKS],
        )
        self.declare_parameter(
            'block_color_bs',
            [color[0] for _name, _bx, _by, _size, color in DEFAULT_SIM_BLOCKS],
        )
        self.declare_parameter(
            'block_color_gs',
            [color[1] for _name, _bx, _by, _size, color in DEFAULT_SIM_BLOCKS],
        )
        self.declare_parameter(
            'block_color_rs',
            [color[2] for _name, _bx, _by, _size, color in DEFAULT_SIM_BLOCKS],
        )

        self.bridge = CvBridge()

        # 相机内参
        self.fx = 500.0
        self.fy = 500.0
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # 相机在基座坐标系中的位姿 (向下看)
        self.cam_pos = np.array([
            self.get_parameter('cam_pos_x').value,
            self.get_parameter('cam_pos_y').value,
            self.get_parameter('cam_pos_z').value,
        ])
        # 旋转矩阵: Xc→+Xb, Yc→-Yb, Zc→-Zb
        self.cam_R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

        self.blocks = build_sim_blocks(
            block_names=self.get_parameter('block_names').value,
            block_xs=self.get_parameter('block_xs').value,
            block_ys=self.get_parameter('block_ys').value,
            block_sizes=self.get_parameter('block_sizes').value,
            block_color_bs=self.get_parameter('block_color_bs').value,
            block_color_gs=self.get_parameter('block_color_gs').value,
            block_color_rs=self.get_parameter('block_color_rs').value,
        )

        self.pub_image = self.create_publisher(Image, '/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera_info', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/block_markers', 10)

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

        # 发布色块 Marker (RViz 可视化)
        ma = MarkerArray()
        for i, (name, bx, by, bsize, bgr) in enumerate(self.blocks):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = msg.header.stamp
            m.ns = 'blocks'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = bx
            m.pose.position.y = by
            m.pose.position.z = bsize / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = bsize
            m.scale.y = bsize
            m.scale.z = bsize
            m.color.r = bgr[2] / 255.0
            m.color.g = bgr[1] / 255.0
            m.color.b = bgr[0] / 255.0
            m.color.a = 0.8
            ma.markers.append(m)
        self.pub_markers.publish(ma)

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
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
