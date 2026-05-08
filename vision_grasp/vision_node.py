#!/usr/bin/env python3
"""视觉识别节点 — HSV 颜色检测 + 轮廓定位，发布检测结果。

使用 geometry_msgs/PoseStamped 发布检测结果:
- position.x/y = 像素坐标 (u, v)
- position.z = 像素宽度
- orientation.x = 像素高度
- orientation.w = 置信度
- header.frame_id = 物体类型
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()

        # HSV 颜色范围定义 (H: 0-179, S: 0-255, V: 0-255)
        self.color_ranges = {
            'red_block': [
                (np.array([0, 120, 100]), np.array([10, 255, 255])),
                (np.array([160, 120, 100]), np.array([179, 255, 255])),
            ],
            'green_block': [
                (np.array([35, 100, 80]), np.array([85, 255, 255])),
            ],
            'blue_block': [
                (np.array([90, 80, 80]), np.array([130, 255, 255])),
            ],
            'yellow_block': [
                (np.array([20, 100, 100]), np.array([35, 255, 255])),
            ],
        }

        self.declare_parameter('min_area', 500)
        self.declare_parameter('erode_iter', 1)
        self.declare_parameter('dilate_iter', 1)

        self.sub_image = self.create_subscription(Image, '/image_raw', self._image_cb, 10)
        self.pub_objects = self.create_publisher(PoseStamped, '/detected_objects', 10)
        self.pub_debug = self.create_publisher(Image, '/debug_image', 10)
        self.get_logger().info('视觉识别节点已启动，等待图像...')

    def _image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        min_area = self.get_parameter('min_area').value
        erode_iter = self.get_parameter('erode_iter').value
        dilate_iter = self.get_parameter('dilate_iter').value
        kernel = np.ones((5, 5), np.uint8)

        debug_img = frame.copy()

        for obj_type, ranges in self.color_ranges.items():
            # 合并多段 HSV 范围 (如红色跨 0°/180°)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # 形态学操作去噪
            if erode_iter > 0:
                mask = cv2.erode(mask, kernel, iterations=erode_iter)
            if dilate_iter > 0:
                mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w / 2.0
                cy = y + h / 2.0

                # 发布检测结果 (用 PoseStamped 编码)
                det = PoseStamped()
                det.header.stamp = self.get_clock().now().to_msg()
                det.header.frame_id = obj_type  # 物体类型放在 frame_id
                det.pose.position.x = cx
                det.pose.position.y = cy
                det.pose.position.z = float(w)  # 像素宽度
                det.pose.orientation.x = float(h)  # 像素高度
                det.pose.orientation.w = min(area / 10000.0, 1.0)  # 置信度
                self.pub_objects.publish(det)

                # 绘制调试图像
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(debug_img, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.putText(debug_img, obj_type, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 发布调试图像
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = msg.header
        self.pub_debug.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
