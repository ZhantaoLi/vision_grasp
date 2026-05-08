#!/usr/bin/env python3
"""trajectory_node 使用的纯函数和状态常量。"""

import math

import numpy as np


ST_IDLE = 0
ST_OPENING = 1
ST_APPROACH = 2
ST_DESCENDING = 3
ST_CLOSING = 4
ST_LIFTING = 5
ST_RETRACTING = 6

GRIPPER7_OPEN = -0.06
GRIPPER8_OPEN = 0.06
GRIPPER_CLOSE_SAFE7 = -0.0025
GRIPPER_CLOSE_SAFE8 = 0.0025
GRIPPER_OFFSET_Z = 0.13503

DEFAULT_BLOCK_WIDTH = 0.035

_NEXT_STATE = {
    ST_OPENING: ST_APPROACH,
    ST_APPROACH: ST_DESCENDING,
    ST_DESCENDING: ST_CLOSING,
    ST_CLOSING: ST_LIFTING,
    ST_LIFTING: ST_RETRACTING,
    ST_RETRACTING: ST_IDLE,
}


def compute_gripper_close_positions(block_width):
    """根据物体宽度计算夹爪闭合位置。"""
    width = block_width if block_width > 0 else DEFAULT_BLOCK_WIDTH
    return -width / 2.0, width / 2.0


def limit_correction_step(error, gain=0.6, max_step_norm=0.03):
    """限制 IK 目标修正的最大步长，避免振荡。"""
    step = gain * error
    step_norm = np.linalg.norm(step)
    if step_norm > max_step_norm:
        step = step * max_step_norm / step_norm
    return step


def compute_motion_duration(start_pos, goal_pos, move_duration, max_vel, tick_period):
    """根据最大关节变化量和速度限制计算实际运动时长。"""
    max_change = max(abs(g - s) for s, g in zip(start_pos, goal_pos))
    vel_duration = max_change / max_vel if max_vel > 0 else 0.0
    return max(move_duration, vel_duration, tick_period)


def interpolate_positions(start_pos, goal_pos, t):
    """对起止关节角做线性插值。"""
    return [s + (g - s) * t for s, g in zip(start_pos, goal_pos)]


def next_state(state):
    """返回抓取状态机的下一个状态。"""
    return _NEXT_STATE.get(state, ST_IDLE)


def format_joint_angles(joint_angles):
    """将弧度角格式化为角度字符串，便于日志输出。"""
    return [f"{math.degrees(angle):.0f}°" for angle in joint_angles]


def state_log_message(state, close7=None, close8=None):
    """返回状态切换时应输出的日志内容。"""
    if state == ST_OPENING:
        return "[1/7] 张开夹爪"
    if state == ST_APPROACH:
        return "[2/7] 移到目标上方"
    if state == ST_DESCENDING:
        return "[3/7] 下降到目标"
    if state == ST_CLOSING:
        return f"[4/7] 闭合夹爪 (j7={close7:+.4f}m, j8={close8:+.4f}m)"
    if state == ST_LIFTING:
        return "[5/7] 抬升机械臂"
    if state == ST_RETRACTING:
        return "[6/7] 归位到安全高度"
    return "[7/7] 抓取完成，等待新目标"
