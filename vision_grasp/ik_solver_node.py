#!/usr/bin/env python3
"""抓取规划节点 — 数值逆运动学 (阻尼最小二乘法)。

Position-only IK，姿态由关节构型自然决定。
URDF 用标准库解析，无外部依赖。
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState


def _rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_matrix(axis, angle):
    k = np.array(axis, dtype=float)
    k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _parse_urdf_joints(urdf_path):
    tree = ET.parse(urdf_path)
    joints = {}
    for j in tree.getroot().findall('joint'):
        name = j.get('name')
        origin = j.find('origin')
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get('xyz'):
                xyz = [float(v) for v in origin.get('xyz').split()]
            if origin.get('rpy'):
                rpy = [float(v) for v in origin.get('rpy').split()]
        ax_el = j.find('axis')
        axis = [0.0, 0.0, 1.0]
        if ax_el is not None and ax_el.get('xyz'):
            axis = [float(v) for v in ax_el.get('xyz').split()]
        joints[name] = {
            'parent': j.find('parent').get('link'),
            'child': j.find('child').get('link'),
            'type': j.get('type', 'fixed'),
            'xyz': xyz, 'rpy': rpy, 'axis': axis,
        }
    return joints


def _find_chain(joints, base, tip):
    c2j = {v['child']: (k, v) for k, v in joints.items()}
    path, cur = [], tip
    for _ in range(20):
        if cur not in c2j:
            break
        jn, jd = c2j[cur]
        path.append((jn, jd))
        cur = jd['parent']
        if cur == base:
            path.reverse()
            return path
    return None


def _fk(q, chain):
    T = np.eye(4)
    for jd, qi in zip(chain, q):
        To = np.eye(4)
        To[:3, :3] = _rpy_to_matrix(*jd['rpy'])
        To[:3, 3] = jd['xyz']
        Tj = np.eye(4)
        Tj[:3, :3] = _axis_angle_matrix(jd['axis'], qi)
        T = T @ To @ Tj
    return T


def _ik_position(target_pos, chain, q_init=None, max_iter=500, tol=1e-4):
    n = len(chain)
    q = np.array(q_init if q_init is not None else [0.0] * n)

    for _ in range(max_iter):
        T = _fk(q, chain)
        err = target_pos - T[:3, 3]
        if np.linalg.norm(err) < tol:
            return q, True

        J = np.zeros((3, n))
        delta = 1e-7
        for j in range(n):
            qd = q.copy()
            qd[j] += delta
            Td = _fk(qd, chain)
            J[:, j] = (Td[:3, 3] - T[:3, 3]) / delta

        dq = J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(3), err)
        q += dq
        q = np.clip(q, -1.57, 1.57)

    T = _fk(q, chain)
    return q, np.linalg.norm(target_pos - T[:3, 3]) < tol * 5


class IKSolverNode(Node):
    JOINT_NAMES = ['Rotation', 'Rotation2', 'Rotation3',
                   'Rotation4', 'Rotation5', 'Rotation6']

    def __init__(self):
        super().__init__('ik_solver_node')
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('base_link', 'Base')
        self.declare_parameter('tip_link', 'zhua')

        self.chain = None
        self._load_kinematics()

        self.sub_target = self.create_subscription(
            PoseStamped, '/grasp_target', self._target_cb, 10)
        self.pub_joints = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_result = self.create_publisher(PoseStamped, '/grasp_result', 10)
        self.get_logger().info('IK 求解节点已启动')

    def _load_kinematics(self):
        urdf_param = self.get_parameter('urdf_file').value
        if urdf_param and os.path.isfile(urdf_param):
            urdf_path = urdf_param
        else:
            try:
                share = get_package_share_directory('vision_grasp')
                urdf_path = os.path.join(share, 'description', 'genkiarm.urdf')
            except Exception:
                urdf_path = os.path.join(os.path.dirname(__file__),
                                         '..', 'description', 'genkiarm.urdf')

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
            q, ok = _ik_position(target, self.chain,
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
        js.name = self.JOINT_NAMES[:len(angles)]
        js.position = angles
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
