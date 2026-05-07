#!/usr/bin/env python3
"""轨迹插值节点 — 状态机控制抓取-抬起完整流程。

状态机:
  IDLE → 开夹爪 → 移到目标 → 闭夹爪 → 抬起 → IDLE

订阅 /joint_goal (IK 关节角度) 和 /grasp_target (目标 3D 坐标)
发布 /joint_states (插值后的关节角度)
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

# ---------- URDF / IK 工具函数 ----------


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
        limit = None
        lim_el = j.find('limit')
        if lim_el is not None:
            limit = {
                'lower': float(lim_el.get('lower', '0')),
                'upper': float(lim_el.get('upper', '0')),
            }
        joints[name] = {
            'parent': j.find('parent').get('link'),
            'child': j.find('child').get('link'),
            'type': j.get('type', 'fixed'),
            'xyz': xyz, 'rpy': rpy, 'axis': axis, 'limit': limit,
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


def _fk_all_links(q, chain):
    """返回每个关节子 link 在基座系中的 z 坐标列表。"""
    T = np.eye(4)
    zs = []
    for jd, qi in zip(chain, q):
        To = np.eye(4)
        To[:3, :3] = _rpy_to_matrix(*jd['rpy'])
        To[:3, 3] = jd['xyz']
        Tj = np.eye(4)
        Tj[:3, :3] = _axis_angle_matrix(jd['axis'], qi)
        T = T @ To @ Tj
        zs.append(T[2, 3])
    return zs


def _ik_position(target_pos, chain, chain_limits=None, q_init=None,
                 max_iter=500, tol=1e-4):
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
        if chain_limits is not None:
            lo = np.array([lim[0] for lim in chain_limits])
            hi = np.array([lim[1] for lim in chain_limits])
            q = np.clip(q, lo, hi)
        else:
            q = np.clip(q, -3.14, 3.14)
    T = _fk(q, chain)
    return q, np.linalg.norm(target_pos - T[:3, 3]) < tol * 5


# ---------- 状态机常量 ----------

ST_IDLE = 0
ST_OPENING = 1
ST_APPROACH = 2
ST_DESCENDING = 3
ST_CLOSING = 4
ST_LIFTING = 5
ST_RETRACTING = 6

GRIPPER_OPEN = -0.06
GRIPPER_CLOSE = 0.0


class TrajectoryNode(Node):
    JOINT_NAMES = ['joint1', 'joint2', 'joint3',
                   'joint4', 'joint5', 'joint6',
                   'joint7', 'joint8']

    def __init__(self):
        super().__init__('trajectory_node')
        self.declare_parameter('move_duration', 3.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('test_mode', False)
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'link6')
        self.declare_parameter('lift_height', 0.2)
        self.declare_parameter('approach_height', 0.15)
        self.declare_parameter('home_height', 0.35)

        self.move_duration = self.get_parameter('move_duration').value
        rate = self.get_parameter('publish_rate').value
        test_mode = self.get_parameter('test_mode').value
        self.lift_height = self.get_parameter('lift_height').value
        self.approach_height = self.get_parameter('approach_height').value
        self.home_height = self.get_parameter('home_height').value

        self._joint_names = list(self.JOINT_NAMES)
        self._current_pos = [0.0] * 8
        self._start_pos = [0.0] * 8
        self._goal_pos = [0.0] * 8
        self._moving = False
        self._start_time = None

        # 状态机
        self._state = ST_IDLE
        self._grasp_joints = None
        self._saved_target = None

        # IK 引擎
        self._ik_chain = None
        self._ik_limits = None
        self._ik_chain_dicts = None
        self._load_ik()

        self.sub_goal = self.create_subscription(
            JointState, '/joint_goal', self._goal_cb, 10)
        self.sub_target = self.create_subscription(
            PoseStamped, '/grasp_target', self._grasp_target_cb, 10)
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)

        self.timer = self.create_timer(1.0 / rate, self._tick)

        if test_mode:
            self._test_poses = [
                [0, 0, 0, 0, 0, 0],
                [30, 45, -30, 0, 0, 0],
                [-30, 60, -45, 30, -20, 0],
                [0, 30, -15, -30, 30, 45],
            ]
            self._test_idx = 0
            self.create_timer(self.move_duration + 1.0, self._test_cycle)
            self.get_logger().info('测试模式: 循环演示各关节运动')
            self._test_cycle()
        else:
            self.get_logger().info(
                f'轨迹节点已启动 (时长 {self.move_duration}s, 频率 {rate}Hz, '
                f'接近 {self.approach_height}m, 抬升 {self.lift_height}m)')

    # ---------- IK 加载 ----------

    def _load_ik(self):
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

        joints = _parse_urdf_joints(urdf_path)
        base = self.get_parameter('base_link').value
        tip = self.get_parameter('tip_link').value
        path = _find_chain(joints, base, tip)
        if path is None:
            self.get_logger().error(f'无法找到 {base} → {tip} 运动链')
            return

        self._ik_chain_dicts = [{'xyz': jd['xyz'], 'rpy': jd['rpy'],
                                 'axis': jd['axis']} for _, jd in path]
        self._ik_limits = []
        for _, jd in path:
            lim = jd.get('limit')
            if lim:
                self._ik_limits.append((lim['lower'], lim['upper']))
            else:
                self._ik_limits.append((-3.14, 3.14))
        self._ik_chain = path
        self.get_logger().info(f'IK 引擎就绪: {len(path)} 关节')

    def _solve_ik(self, target_pos):
        if self._ik_chain_dicts is None:
            return None
        best_q, best_err = None, float('inf')
        for g_deg in [0, 30, -30, 45, -45, 60, -60]:
            q, ok = _ik_position(
                target_pos, self._ik_chain_dicts, self._ik_limits,
                q_init=[math.radians(g_deg)] * len(self._ik_chain_dicts))
            T = _fk(q, self._ik_chain_dicts)
            err = np.linalg.norm(target_pos - T[:3, 3])
            if ok and err < best_err:
                best_q, best_err = q, err
                if err < 1e-4:
                    break
        return best_q

    # ---------- 测试模式 ----------

    def _test_cycle(self):
        pose = self._test_poses[self._test_idx]
        self._start_pos = list(self._current_pos)
        self._goal_pos = [math.radians(a) for a in pose] + \
                         [GRIPPER_OPEN, GRIPPER_OPEN]
        self._joint_names = list(self.JOINT_NAMES)
        self._moving = True
        self._start_time = self.get_clock().now()
        self.get_logger().info(f'测试姿态 {self._test_idx}: {pose}')
        self._test_idx = (self._test_idx + 1) % len(self._test_poses)

    # ---------- /joint_goal 回调 (忽略，防止干扰状态机) ----------

    def _goal_cb(self, msg: JointState):
        pass  # 完全由状态机控制，忽略外部关节目标

    # ---------- /grasp_target 回调 (入口) ----------

    def _grasp_target_cb(self, msg: PoseStamped):
        if self._state != ST_IDLE:
            self.get_logger().debug('正在执行抓取，忽略新目标')
            return
        if self._ik_chain_dicts is None:
            return

        p = msg.pose.position
        target = np.array([p.x, p.y, p.z])
        self._saved_target = target.copy()

        self.get_logger().info(
            f'收到目标: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

        # 计算 IK: 目标位置
        q_grasp = self._solve_ik(target)
        if q_grasp is None:
            self.get_logger().warn('目标 IK 失败')
            return

        # 计算 IK: 接近位置，自动抬高直到下降路径安全
        q_approach = None
        approach_h = self.approach_height
        for _ in range(5):
            approach_target = target.copy()
            approach_target[2] += approach_h
            q_try = self._solve_ik(approach_target)
            if q_try is None:
                approach_h += 0.1
                continue
            # 采样中点检查所有 link 的 z 坐标
            q_mid = (np.array(q_try) + np.array(q_grasp)) / 2.0
            zs = _fk_all_links(q_mid, self._ik_chain_dicts)
            if all(z > 0.02 for z in zs):
                q_approach = q_try
                break
            self.get_logger().info(
                f'下降路径不安全 (最低 z={min(zs):.3f})，'
                f'抬高接近高度到 {approach_h + 0.1:.2f}m')
            approach_h += 0.1

        if q_approach is None:
            self.get_logger().warn('无法找到安全的接近位置')
            return

        # 计算 IK: 抬升位置 (z + lift_height)
        lift_target = target.copy()
        lift_target[2] += self.lift_height
        q_lift = self._solve_ik(lift_target)
        if q_lift is None:
            self.get_logger().warn('抬升 IK 失败，使用目标位置')
            q_lift = q_grasp

        # 计算 IK: 归位位置 (高于抬升, 跨目标移动安全)
        home_target = target.copy()
        home_target[2] = self.home_height
        q_home = self._solve_ik(home_target)
        if q_home is None:
            self.get_logger().warn('归位 IK 失败，使用抬升位置')
            q_home = q_lift

        self._approach_joints = list(q_approach) + [GRIPPER_OPEN, GRIPPER_OPEN]
        self._grasp_joints = list(q_grasp) + [GRIPPER_OPEN, GRIPPER_OPEN]
        self._lift_joints = list(q_lift) + [GRIPPER_CLOSE, GRIPPER_CLOSE]
        self._home_joints = list(q_home) + [GRIPPER_CLOSE, GRIPPER_CLOSE]

        self.get_logger().info(
            f'接近关节: {[f"{math.degrees(a):.0f}°" for a in q_approach]}')
        self.get_logger().info(
            f'抓取关节: {[f"{math.degrees(a):.0f}°" for a in q_grasp]}')
        self.get_logger().info(
            f'抬升关节: {[f"{math.degrees(a):.0f}°" for a in q_lift]}')
        self.get_logger().info(
            f'归位关节: {[f"{math.degrees(a):.0f}°" for a in q_home]}')

        # 开始状态机: 张开夹爪
        self._enter_state(ST_OPENING)

    # ---------- 状态机 ----------

    def _enter_state(self, state):
        self._state = state

        if state == ST_OPENING:
            self.get_logger().info('[1/7] 张开夹爪')
            self._set_arm_goal(self._current_pos, GRIPPER_OPEN, GRIPPER_OPEN)

        elif state == ST_APPROACH:
            self.get_logger().info('[2/7] 移到目标上方')
            self._set_arm_goal(self._approach_joints)

        elif state == ST_DESCENDING:
            self.get_logger().info('[3/7] 下降到目标')
            self._set_arm_goal(self._grasp_joints)

        elif state == ST_CLOSING:
            self.get_logger().info('[4/7] 闭合夹爪')
            self._set_arm_goal(self._current_pos, GRIPPER_CLOSE, GRIPPER_CLOSE)

        elif state == ST_LIFTING:
            self.get_logger().info('[5/7] 抬升机械臂')
            self._set_arm_goal(self._lift_joints)

        elif state == ST_RETRACTING:
            self.get_logger().info('[6/7] 归位到安全高度')
            self._set_arm_goal(self._home_joints)

        elif state == ST_IDLE:
            self.get_logger().info('[7/7] 抓取完成，等待新目标')
            self._grasp_joints = None

    def _set_arm_goal(self, goal, gripper7=None, gripper8=None):
        self._start_pos = list(self._current_pos)
        self._goal_pos = list(goal)
        if gripper7 is not None:
            if 'joint7' in self._joint_names:
                idx7 = self._joint_names.index('joint7')
                self._goal_pos[idx7] = gripper7
        if gripper8 is not None:
            if 'joint8' in self._joint_names:
                idx8 = self._joint_names.index('joint8')
                self._goal_pos[idx8] = gripper8
        while len(self._start_pos) < len(self._goal_pos):
            self._start_pos.append(0.0)
        self._start_pos = self._start_pos[:len(self._goal_pos)]
        self._moving = True
        self._start_time = self.get_clock().now()

    # ---------- 定时器回调 ----------

    def _tick(self):
        if not self._moving:
            return

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
        t = min(elapsed / self.move_duration, 1.0)

        pos = [s + (g - s) * t for s, g in zip(self._start_pos, self._goal_pos)]
        self._current_pos = list(pos)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self._joint_names
        js.position = pos
        self.pub_js.publish(js)

        if t >= 1.0:
            self._moving = False
            self.get_logger().info(
                f'阶段完成 (state={self._state}), t={t:.2f}')
            # 状态转换
            if self._state == ST_OPENING:
                self._enter_state(ST_APPROACH)
            elif self._state == ST_APPROACH:
                self._enter_state(ST_DESCENDING)
            elif self._state == ST_DESCENDING:
                self._enter_state(ST_CLOSING)
            elif self._state == ST_CLOSING:
                self._enter_state(ST_LIFTING)
            elif self._state == ST_LIFTING:
                self._enter_state(ST_RETRACTING)
            elif self._state == ST_RETRACTING:
                self._enter_state(ST_IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
