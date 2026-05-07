#!/usr/bin/env python3
"""驱动执行节点 — 订阅 /joint_states，可选驱动真实舵机。"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


SERVO_MAP = {
    'joint1': 1, 'joint2': 2, 'joint3': 3,
    'joint4': 4, 'joint5': 5, 'joint6': 6,
}


class ArmDriverNode(Node):
    def __init__(self):
        super().__init__('arm_driver_node')
        self.declare_parameter('use_serial', False)
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 1000000)

        self.serial_ctrl = None
        self._init_serial_if_needed()

        self.sub_js = self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)
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

    def _js_cb(self, msg: JointState):
        if self.serial_ctrl is None:
            return
        for name, pos in zip(msg.name, msg.position):
            sid = SERVO_MAP.get(name)
            if sid is not None:
                deg = math.degrees(pos)
                self.serial_ctrl.set_angle(sid, deg)

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
