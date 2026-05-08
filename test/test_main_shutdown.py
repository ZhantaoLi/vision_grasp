from importlib import import_module

import pytest


MODULE_CASES = [
    ("vision_grasp.camera_node", "CameraNode"),
    ("vision_grasp.vision_node", "VisionNode"),
    ("vision_grasp.tf_transformer_node", "TfTransformerNode"),
    ("vision_grasp.trajectory_node", "TrajectoryNode"),
    ("vision_grasp.arm_driver_node", "ArmDriverNode"),
]


@pytest.mark.parametrize(("module_name", "class_name"), MODULE_CASES)
def test_main_handles_interrupt_when_context_is_already_shutdown(
    monkeypatch, module_name, class_name
):
    module = import_module(module_name)
    state = {"context_ok": False, "destroy_called": False}

    class FakeNode:
        def destroy_node(self):
            state["destroy_called"] = True

    def fake_init(args=None):
        state["context_ok"] = True

    def fake_spin(_node):
        state["context_ok"] = False
        raise KeyboardInterrupt()

    def fake_shutdown():
        if not state["context_ok"]:
            raise RuntimeError("context already shutdown")
        state["context_ok"] = False

    def fake_try_shutdown():
        state["context_ok"] = False

    monkeypatch.setattr(module, class_name, FakeNode)
    monkeypatch.setattr(module.rclpy, "init", fake_init)
    monkeypatch.setattr(module.rclpy, "spin", fake_spin)
    monkeypatch.setattr(module.rclpy, "shutdown", fake_shutdown)
    monkeypatch.setattr(module.rclpy, "try_shutdown", fake_try_shutdown, raising=False)

    module.main()

    assert state["destroy_called"], "main() should destroy the node during cleanup"
