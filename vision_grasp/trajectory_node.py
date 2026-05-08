#!/usr/bin/env python3
"""轨迹插值节点 — 状态机控制抓取-抬起完整流程。

状态机:
  IDLE → 开夹爪 → 移到目标 → 闭夹爪 → 抬起 → IDLE

订阅 /grasp_target (目标 3D 坐标 + block 宽度)
发布 /joint_states (插值后的关节角度)
"""

import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from .ik_utils import (_fk, _fk_all_links, _ik_position,
                       _parse_urdf_joints, _find_chain)
from .trajectory_support import (
    GRIPPER7_OPEN,
    GRIPPER8_OPEN,
    GRIPPER_CLOSE_SAFE7,
    GRIPPER_CLOSE_SAFE8,
    GRIPPER_OFFSET_Z,
    ST_APPROACH,
    ST_CLOSING,
    ST_DESCENDING,
    ST_IDLE,
    ST_LIFTING,
    ST_OPENING,
    ST_RETRACTING,
    compute_gripper_close_positions,
    compute_motion_duration,
    format_joint_angles,
    interpolate_positions,
    limit_correction_step,
    next_state,
    state_log_message,
)


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
        self.declare_parameter('max_joint_velocity', 1.0)

        self.move_duration = self.get_parameter('move_duration').value
        rate = self.get_parameter('publish_rate').value
        test_mode = self.get_parameter('test_mode').value
        self.lift_height = self.get_parameter('lift_height').value
        self.approach_height = self.get_parameter('approach_height').value
        self.home_height = self.get_parameter('home_height').value
        self.max_vel = self.get_parameter('max_joint_velocity').value
        self._tick_period = 1.0 / rate

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
        self._gripper_close7 = GRIPPER_CLOSE_SAFE7
        self._gripper_close8 = GRIPPER_CLOSE_SAFE8

        # IK 引擎
        self._ik_chain = None
        self._ik_limits = None
        self._ik_chain_dicts = None
        self._load_ik()

        self.sub_target = self.create_subscription(
            PoseStamped, '/grasp_target', self._grasp_target_cb, 10)
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_marker = self.create_publisher(Marker, '/grasp_marker', 10)

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

    def _gripper_z(self, q):
        """计算夹爪原点在基座系中的 z 坐标。"""
        T = _fk(q, self._ik_chain_dicts)
        # 夹爪沿 link6 z 轴偏移 GRIPPER_OFFSET_Z
        return T[2, 3] + T[2, 2] * GRIPPER_OFFSET_Z

    def _gripper_tip(self, q):
        """计算夹爪原点在基座系中的完整 3D 坐标。"""
        T = _fk(q, self._ik_chain_dicts)
        # 夹爪沿 link6 z 轴偏移 GRIPPER_OFFSET_Z（XYZ 三个方向都有分量）
        return T[:3, 3] + T[:3, 2] * GRIPPER_OFFSET_Z

    # ---------- 测试模式 ----------

    def _test_cycle(self):
        pose = self._test_poses[self._test_idx]
        self._start_pos = list(self._current_pos)
        self._goal_pos = [math.radians(a) for a in pose] + \
                         [GRIPPER7_OPEN, GRIPPER8_OPEN]
        self._joint_names = list(self.JOINT_NAMES)
        self._moving = True
        self._start_time = self.get_clock().now()
        self.get_logger().info(f'测试姿态 {self._test_idx}: {pose}')
        self._test_idx = (self._test_idx + 1) % len(self._test_poses)

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

        # 根据 block 宽度计算夹爪闭合位置
        # joint7 沿 -X, joint8 沿 +X: 需要相反符号才能从两侧夹紧
        block_w = msg.pose.orientation.x if msg.pose.orientation.x > 0 else 0.035
        self._gripper_close7, self._gripper_close8 = (
            compute_gripper_close_positions(block_w)
        )
        self.get_logger().info(
            f'block 宽度 {block_w:.3f}m → 夹爪 j7={self._gripper_close7:+.4f}m, '
            f'j8={self._gripper_close8:+.4f}m')

        self.get_logger().info(
            f'收到目标: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

        # 发布目标点 Marker (RViz 可视化)
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'grasp_target'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = p.x
        m.pose.position.y = p.y
        m.pose.position.z = p.z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.9
        self.pub_marker.publish(m)

        # 计算 IK: 迭代纠正夹爪 XYZ 偏移 (阻尼修正)
        q_grasp = None
        best_q, best_err = None, float('inf')
        grasp_target = target.copy()
        for it in range(20):
            q_try = self._solve_ik(grasp_target)
            if q_try is None:
                break
            tip = self._gripper_tip(q_try)
            gz = tip[2]
            if gz < 0.005:
                old_z = grasp_target[2]
                grasp_target[2] += -gz + 0.01
                self.get_logger().info(
                    f'夹爪穿地 (z={gz:.3f}m)，抬高目标 '
                    f'{old_z:.3f} → {grasp_target[2]:.3f}m')
            err = target - tip
            err_norm = np.linalg.norm(err)
            self.get_logger().debug(
                f'[{it}] 夹爪: ({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) '
                f'偏差: {err_norm:.4f}m')
            if err_norm < best_err:
                best_q, best_err = q_try, err_norm
            if err_norm < 0.005:
                q_grasp = q_try
                break
            # 阻尼修正: 每次只修正部分误差，限制最大步长避免振荡
            grasp_target = grasp_target + limit_correction_step(err)
        if q_grasp is None:
            if best_q is not None and best_err < 0.02:
                q_grasp = best_q
                self.get_logger().info(
                    f'使用最佳解 (偏差 {best_err:.4f}m)')
            else:
                self.get_logger().warn(
                    f'目标 IK 失败 (最佳偏差 {best_err:.4f}m)')
                return

        # 计算 IK: 接近位置，自动抬高直到下降路径安全
        q_approach = None
        approach_h = self.approach_height
        for _ in range(5):
            approach_target = grasp_target.copy()
            approach_target[2] += approach_h
            q_try = self._solve_ik(approach_target)
            if q_try is None:
                approach_h += 0.1
                continue
            # 采样多个点检查所有 link 和夹爪的 z 坐标
            safe = True
            for t_check in [0.25, 0.5, 0.75]:
                q_check = (np.array(q_try) * (1 - t_check)
                           + np.array(q_grasp) * t_check)
                zs = _fk_all_links(q_check, self._ik_chain_dicts)
                gz = self._gripper_z(q_check)
                all_z = zs + [gz]
                if any(z <= 0.02 for z in all_z):
                    safe = False
                    break
            if safe:
                q_approach = q_try
                break
            self.get_logger().info(
                f'下降路径不安全 (最低 z={min(all_z):.3f})，'
                f'抬高接近高度到 {approach_h + 0.1:.2f}m')
            approach_h += 0.1

        if q_approach is None:
            self.get_logger().warn('无法找到安全的接近位置')
            return

        # 计算 IK: 抬升位置 (z + lift_height)
        lift_target = grasp_target.copy()
        lift_target[2] += self.lift_height
        q_lift = self._solve_ik(lift_target)
        if q_lift is None:
            self.get_logger().warn('抬升 IK 失败，使用目标位置')
            q_lift = q_grasp

        # 计算 IK: 归位位置 (高于抬升, 跨目标移动安全)
        home_target = grasp_target.copy()
        home_target[2] = self.home_height
        q_home = self._solve_ik(home_target)
        if q_home is None:
            self.get_logger().warn('归位 IK 失败，使用抬升位置')
            q_home = q_lift

        gc7, gc8 = self._gripper_close7, self._gripper_close8
        self._approach_joints = list(q_approach) + [GRIPPER7_OPEN, GRIPPER8_OPEN]
        self._grasp_joints = list(q_grasp) + [GRIPPER7_OPEN, GRIPPER8_OPEN]
        self._lift_joints = list(q_lift) + [gc7, gc8]
        self._home_joints = list(q_home) + [gc7, gc8]

        self.get_logger().info(
            f'接近关节: {format_joint_angles(q_approach)}')
        self.get_logger().info(
            f'抓取关节: {format_joint_angles(q_grasp)}')
        self.get_logger().info(
            f'抬升关节: {format_joint_angles(q_lift)}')
        self.get_logger().info(
            f'归位关节: {format_joint_angles(q_home)}')

        # 开始状态机: 张开夹爪
        self._enter_state(ST_OPENING)

    # ---------- 状态机 ----------

    def _enter_state(self, state):
        self._state = state
        self.get_logger().info(
            state_log_message(state, self._gripper_close7, self._gripper_close8)
        )

        if state == ST_OPENING:
            self._set_arm_goal(self._current_pos, GRIPPER7_OPEN, GRIPPER8_OPEN)

        elif state == ST_APPROACH:
            self._set_arm_goal(self._approach_joints)

        elif state == ST_DESCENDING:
            self._set_arm_goal(self._grasp_joints)

        elif state == ST_CLOSING:
            self._set_arm_goal(
                self._current_pos, self._gripper_close7, self._gripper_close8
            )

        elif state == ST_LIFTING:
            self._set_arm_goal(self._lift_joints)

        elif state == ST_RETRACTING:
            self._set_arm_goal(self._home_joints)

        elif state == ST_IDLE:
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

        # 根据关节速度限制计算实际运动时长
        self._actual_duration = compute_motion_duration(
            start_pos=self._start_pos,
            goal_pos=self._goal_pos,
            move_duration=self.move_duration,
            max_vel=self.max_vel,
            tick_period=self._tick_period,
        )

        self._moving = True
        self._start_time = self.get_clock().now()

    # ---------- 定时器回调 ----------

    def _tick(self):
        if not self._moving:
            return

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
        t = min(elapsed / self._actual_duration, 1.0)

        pos = interpolate_positions(self._start_pos, self._goal_pos, t)
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
            self._enter_state(next_state(self._state))


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
