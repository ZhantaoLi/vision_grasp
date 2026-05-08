from pathlib import Path

import numpy as np
import pytest

from vision_grasp.ik_utils import _find_chain, _fk, _ik_position, _parse_urdf_joints


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PACKAGE_ROOT / "description" / "piper.urdf"


def test_parse_urdf_exposes_expected_gripper_limits():
    joints = _parse_urdf_joints(str(URDF_PATH))
    assert joints["joint7"]["limit"]["lower"] == pytest.approx(-0.08)
    assert joints["joint7"]["limit"]["upper"] == pytest.approx(0.0)
    assert joints["joint8"]["limit"]["lower"] == pytest.approx(0.0)
    assert joints["joint8"]["limit"]["upper"] == pytest.approx(0.08)


def test_find_chain_returns_six_arm_joints_for_link6():
    joints = _parse_urdf_joints(str(URDF_PATH))
    chain = _find_chain(joints, "base_link", "link6")
    assert chain is not None
    assert [joint_name for joint_name, _joint in chain] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]


def test_ik_position_reaches_a_simple_two_joint_target():
    chain = [
        {"xyz": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]},
        {"xyz": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0]},
    ]
    target = np.array([1.0, 1.0, 0.0])

    solution, ok = _ik_position(
        target,
        chain,
        chain_limits=[(-np.pi, np.pi), (-np.pi, np.pi)],
    )

    assert ok
    transform = _fk(solution, chain)
    assert np.allclose(transform[:3, 3], target, atol=1e-3)
