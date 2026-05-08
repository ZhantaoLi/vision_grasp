import math

import numpy as np

from vision_grasp import trajectory_support


def test_compute_gripper_close_positions_uses_opposite_signs():
    close7, close8 = trajectory_support.compute_gripper_close_positions(0.035)
    assert close7 == -0.0175
    assert close8 == 0.0175


def test_compute_motion_duration_respects_velocity_limit():
    duration = trajectory_support.compute_motion_duration(
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.5, 0.0],
        move_duration=1.0,
        max_vel=0.25,
        tick_period=0.05,
    )
    assert duration == 4.0


def test_limit_correction_step_clamps_large_updates():
    step = trajectory_support.limit_correction_step(
        error=np.array([1.0, 0.0, 0.0]),
        gain=0.6,
        max_step_norm=0.03,
    )
    assert np.allclose(step, np.array([0.03, 0.0, 0.0]))


def test_interpolate_positions_returns_linear_blend():
    positions = trajectory_support.interpolate_positions(
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, -1.0, 2.0],
        t=0.25,
    )
    assert positions == [0.25, -0.25, 0.5]


def test_next_state_advances_until_idle():
    state = trajectory_support.ST_OPENING
    visited = [state]
    while state != trajectory_support.ST_IDLE:
        state = trajectory_support.next_state(state)
        visited.append(state)
    assert visited == [
        trajectory_support.ST_OPENING,
        trajectory_support.ST_APPROACH,
        trajectory_support.ST_DESCENDING,
        trajectory_support.ST_CLOSING,
        trajectory_support.ST_LIFTING,
        trajectory_support.ST_RETRACTING,
        trajectory_support.ST_IDLE,
    ]


def test_format_joint_angles_returns_degree_labels():
    labels = trajectory_support.format_joint_angles([0.0, math.pi / 2, -math.pi / 4])
    assert labels == ["0°", "90°", "-45°"]
