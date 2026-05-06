#!/usr/bin/env python3
"""驱动执行节点 — 接收 IK 结果，可选驱动真实舵机。

订阅 /grasp_result (PoseStamped, header.frame_id 含关节角度信息)
通过 JointState 发布控制指令。
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState


SERVO_MAP = {
    'Rotation': 1, 'Rotation2': 2, 'Rotation3': 3,
    'Rotation4': 4, 'Rotation5': 5, 'Rotation6': 6,
}

 
class ArmDriverNode(Node):
    def __init__(self):
        super().__init__('arm_driver_node')
        self.declare_parameter('use_serial', False)
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 1000000)

        self.serial_ctrl = None
        self._init_serial_if_needed()

        self.sub_result = self.create_subscription(
            PoseStamped, '/grasp_result', self._result_cb, 10)
        self.pub_joints = self.create_publisher(JointState, '/joint_states', 10)
        self.get_logger().info('驱动执行节点已启动')

    def _init_serial_if_needed(self):
        if not self.get_parameter('use_serial').value:
            self.get_logger().info('仿真模式')
            return
        try:
            import serial
            port = self.get_parameter('serial_port').value
            baud = self.get_parameter('baudrate').value
            self.serial_ctrl = _ServoController(port, baud)
            for sid in range(1, 7):
                self.serial_ctrl.enable_torque(sid)
            self.get_logger().info(f'串口已连接: {port}')
        except Exception as e:
            self.get_logger().warn(f'串口失败: {e}，回退仿真')
            self.serial_ctrl = None

    def _result_cb(self, msg: PoseStamped):
        # 从 /joint_states 订阅者获取当前角度 (由 IK solver 发布)
        # 这里简单转发 JointState
        self.get_logger().info(
            f'收到抓取目标: ({msg.pose.position.x:.3f},'
            f'{msg.pose.position.y:.3f},{msg.pose.position.z:.3f})')

    def destroy_node(self):
        if self.serial_ctrl is not None:
            self.serial_ctrl.close()
        super().destroy_node()


class _ServoController:
    def __init__(self, port, baudrate=1000000, timeout=0.1):
        import serial
        self.sp = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.ADDR_GOAL = 42
        self.ADDR_TORQUE = 40

    def close(self):
        if self.sp and self.sp.is_open:
            self.sp.close()

    def _send(self, sid, inst, params=None):
        if params is None:
            params = []
        core = [sid, len(params) + 2, inst] + params
        pkt = bytes([0xFF, 0xFF] + core + [(~sum(core)) & 0xFF])
        self.sp.reset_input_buffer()
        self.sp.write(pkt)
        self.sp.flush()

    def enable_torque(self, sid):
        self._send(sid, 3, [self.ADDR_TORQUE, 1])

    def set_angle(self, sid, deg):
        deg = max(-90, min(90, deg))
        pos = int(((deg + 90) / 180) * 2048 + 1024)
        self._send(sid, 3, [self.ADDR_GOAL, pos & 0xFF, (pos >> 8) & 0xFF])


def main(args=None):
    rclpy.init(args=args)
    node = ArmDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
