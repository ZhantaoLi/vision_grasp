#!/usr/bin/env python3
"""抓取规划节点 — 数值逆运动学 (阻尼最小二乘法)。

Position-only IK，姿态由关节构型自然决定。
URDF 用标准库解析，无外部依赖。
"""

import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .ik_utils import _fk, _ik_position, _parse_urdf_joints, _find_chain


class IKSolverNode(Node):
    JOINT_NAMES = ['joint1', 'joint2', 'joint3',
                   'joint4', 'joint5', 'joint6']
    GRIPPER_JOINTS = ['joint7', 'joint8']
    JOINT_LIMITS = [
        (-2.618, 2.618), (0, 3.14), (-2.967, 0),
        (-1.832, 1.832), (-1.22, 1.22), (-3.14, 3.14),
    ]

    def __init__(self):
        super().__init__('ik_solver_node')
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'link6')

        self.chain = None
        self.chain_limits = None
        self._load_kinematics()

        self.sub_target = self.create_subscription(
            PoseStamped, '/grasp_target', self._target_cb, 10)
        self.pub_joints = self.create_publisher(JointState, '/joint_goal', 10)
        self.pub_result = self.create_publisher(PoseStamped, '/grasp_result', 10)
        self.get_logger().info('IK 求解节点已启动')

    def _load_kinematics(self):
        urdf_param = self.get_parameter('urdf_file').value
        if urdf_param and os.path.isfile(urdf_param):
            urdf_path = urdf_param
        else:
            try:
                share = get_package_share_directory('vision_grasp')
                urdf_path = os.path.join(share, 'description', 'piper.urdf')
            except Exception:
                urdf_path = os.path.join(os.path.dirname(__file__),
                                         '..', 'description', 'piper.urdf')

        self.get_logger().info(f'加载 URDF: {urdf_path}')
        joints = _parse_urdf_joints(urdf_path)
        base = self.get_parameter('base_link').value
        tip = self.get_parameter('tip_link').value
        path = _find_chain(joints, base, tip)
        if path is None:
            self.get_logger().error(f'无法找到 {base} → {tip} 运动链')
            return
        self.chain = [{'xyz': jd['xyz'], 'rpy': jd['rpy'], 'axis': jd['axis']}
                      for _, jd in path]
        self.chain_limits = []
        for jn, jd in path:
            lim = jd.get('limit')
            if lim:
                self.chain_limits.append((lim['lower'], lim['upper']))
            else:
                self.chain_limits.append((-3.14, 3.14))
        T0 = _fk([0] * len(self.chain), self.chain)
        self.get_logger().info(
            f'运动链: {len(self.chain)} 关节, '
            f'FK(zero)=({T0[0,3]:.3f},{T0[1,3]:.3f},{T0[2,3]:.3f})')

    def _target_cb(self, msg: PoseStamped):
        if self.chain is None:
            return

        p = msg.pose.position
        target = np.array([p.x, p.y, p.z])

        best_q, best_err = None, float('inf')
        for g_deg in [0, 30, -30, 45, -45, 60, -60]:
            q, ok = _ik_position(target, self.chain, self.chain_limits,
                                 q_init=[math.radians(g_deg)] * len(self.chain))
            T = _fk(q, self.chain)
            err = np.linalg.norm(target - T[:3, 3])
            if ok and err < best_err:
                best_q, best_err = q, err
                if err < 1e-4:
                    break

        if best_q is None:
            self.get_logger().warn(
                f'IK 失败: ({p.x:.3f},{p.y:.3f},{p.z:.3f})')
            return

        angles = best_q.tolist()
        self.get_logger().info(
            f'IK 成功: angles={[f"{math.degrees(a):.1f}" for a in angles]}, '
            f'err={best_err*1000:.1f}mm')

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.JOINT_NAMES[:len(angles)] + self.GRIPPER_JOINTS
        js.position = angles + [0.0, 0.0]
        self.pub_joints.publish(js)

        result = PoseStamped()
        result.header = msg.header
        result.pose.position = msg.pose.position
        result.pose.orientation.w = 1.0
        self.pub_result.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = IKSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
