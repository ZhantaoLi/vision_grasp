# vision_grasp

`vision_grasp` 是一个基于 ROS 2 Jazzy 的视觉抓取原型包，当前重点是跑通一条完整的仿真链路：

1. `camera_node` 生成测试图像和相机信息
2. `vision_node` 做颜色分割和目标检测
3. `tf_transformer_node` 将像素目标转换到机械臂基座坐标系
4. `trajectory_node` 做 IK 求解、轨迹插值和抓取状态机
5. `arm_driver_node` 订阅关节命令，默认以仿真模式运行

当前状态：

- 仿真链路已经可运行
- 默认模式不连接真实相机，也不连接真实舵机
- 真机模式保留了参数入口，但仍需按具体硬件环境做联调

## Directory Layout

```text
vision_grasp/
├── config/
│   ├── params.yaml
│   └── vision_grasp.rviz
├── doc/
│   ├── architecture.md
│   └── architecture_diagram.html
├── description/
│   ├── meshes/
│   └── piper.urdf
├── launch/
│   ├── demo.launch.py
│   └── pipeline.launch.py
├── vision_grasp/
│   ├── arm_driver_node.py
│   ├── camera_node.py
│   ├── ik_utils.py
│   ├── trajectory_support.py
│   ├── tf_transformer_node.py
│   ├── trajectory_node.py
│   └── vision_node.py
├── package.xml
├── resource/vision_grasp
├── setup.cfg
└── setup.py
```

## Dependencies

基础环境：

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3.12

常用运行依赖：

- `ros-jazzy-cv-bridge`
- `ros-jazzy-tf2-ros`
- `ros-jazzy-robot-state-publisher`
- `ros-jazzy-rviz2`
- `python3-opencv`
- `python3-numpy`
- `python3-serial`（仅真机串口模式需要）

示例安装：

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-cv-bridge \
  ros-jazzy-tf2-ros \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-rviz2 \
  python3-opencv \
  python3-numpy \
  python3-serial
```

## Build

在工作区根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vision_grasp --symlink-install
source install/setup.bash
```

## Quick Start

推荐先跑纯仿真管线：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch vision_grasp pipeline.launch.py
```

如果希望同时打开 RViz：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch vision_grasp demo.launch.py
```

两者区别：

- `pipeline.launch.py`：启动抓取管线，不打开 RViz
- `demo.launch.py`：在 `pipeline` 基础上额外启动 `rviz2`

## Architecture Docs

正式架构文档位于：

- [doc/architecture.md](./doc/architecture.md)
- [doc/architecture_diagram.html](./doc/architecture_diagram.html)

## Simulation Mode

默认配置位于 [config/params.yaml](./config/params.yaml)。

当前默认仿真参数：

- `camera_node.use_camera: false`
- `arm_driver_node.use_serial: false`

这意味着：

- 图像由 `camera_node` 直接生成，不依赖 USB 摄像头
- `arm_driver_node` 不会打开串口，不会向真实舵机发送命令
- `robot_state_publisher` 会发布机械臂模型
- `trajectory_node` 会发布 `/joint_states`
- RViz 中可以看到机械臂模型、色块 marker 和抓取目标 marker

仿真场景中的色块现在由 [config/params.yaml](./config/params.yaml) 参数驱动，不再写死在 Python 代码里。改测试场景时只需要改参数文件，然后重新 launch。

当前默认场景参数：

```yaml
camera_node:
  ros__parameters:
    block_names: ['red_block', 'green_block', 'blue_block', 'yellow_block']
    block_xs: [0.30, 0.00, 0.35, 0.05]
    block_ys: [0.15, -0.18, -0.12, 0.20]
    block_sizes: [0.035, 0.035, 0.030, 0.032]
    block_color_bs: [0, 0, 200, 0]
    block_color_gs: [0, 180, 100, 200]
    block_color_rs: [200, 0, 0, 220]
```

这些数组按索引一一对应；同一索引组成一个色块配置。长度必须一致。

## Hardware Mode

真机模式需要至少两类外设：

- 相机
- 机械臂串口控制链路

对应参数在 [config/params.yaml](./config/params.yaml) 中：

```yaml
camera_node:
  ros__parameters:
    use_camera: true
    camera_id: 0

arm_driver_node:
  ros__parameters:
    use_serial: true
    serial_port: /dev/ttyUSB0
    baudrate: 1000000
```

注意：

- `use_camera: true` 后，`camera_node` 会尝试通过 OpenCV 打开真实摄像头
- `use_serial: true` 后，`arm_driver_node` 会尝试打开串口并向舵机发送控制命令
- 当前包更偏原型验证，真机模式默认不保证开箱即用，通常还需要根据实际舵机 ID、串口权限、机械零位和安全限位做现场校准

## Nodes

### `camera_node`

职责：

- 发布 `/image_raw`
- 发布 `/camera_info`
- 发布 `/block_markers`
- 在仿真模式下直接生成带彩色方块的测试图像

### `vision_node`

职责：

- 订阅 `/image_raw`
- 发布 `/detected_objects`
- 发布 `/debug_image`

实现方式：

- 基于 HSV 颜色阈值做红、绿、蓝、黄方块检测
- 输出当前检测框的中心和尺寸

### `tf_transformer_node`

职责：

- 订阅 `/camera_info`
- 订阅 `/detected_objects`
- 发布 `/grasp_target`

实现方式：

- 使用相机内参和固定外参
- 通过射线与 `z=0` 平面求交，将像素坐标映射到 `base_link`

### `trajectory_node`

职责：

- 订阅 `/grasp_target`
- 发布 `/joint_states`
- 发布 `/grasp_marker`

实现方式：

- 读取 `piper.urdf`
- 使用 `ik_utils.py` 做正逆运动学计算
- 执行开夹爪、接近、下降、闭合、抬升、归位的抓取状态机

### `arm_driver_node`

职责：

- 订阅 `/joint_states`
- 在真机模式下通过串口向舵机下发角度命令

默认行为：

- `use_serial: false` 时只打印仿真模式日志，不驱动真实硬件

## Topics

主要话题如下：

| Topic | Type | Producer | Consumer |
|---|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | `camera_node` | `vision_node` |
| `/camera_info` | `sensor_msgs/CameraInfo` | `camera_node` | `tf_transformer_node` |
| `/block_markers` | `visualization_msgs/MarkerArray` | `camera_node` | RViz |
| `/detected_objects` | `geometry_msgs/PoseStamped` | `vision_node` | `tf_transformer_node` |
| `/debug_image` | `sensor_msgs/Image` | `vision_node` | 调试可视化 |
| `/grasp_target` | `geometry_msgs/PoseStamped` | `tf_transformer_node` | `trajectory_node` |
| `/joint_states` | `sensor_msgs/JointState` | `trajectory_node` | `arm_driver_node` / `robot_state_publisher` |
| `/grasp_marker` | `visualization_msgs/Marker` | `trajectory_node` | RViz |

## Parameters

参数文件见 [config/params.yaml](./config/params.yaml)。

当前主要参数：

### `camera_node`

- `use_camera`
- `camera_id`
- `fps`
- `width`
- `height`
- `cam_pos_x`
- `cam_pos_y`
- `cam_pos_z`
- `block_names`
- `block_xs`
- `block_ys`
- `block_sizes`
- `block_color_bs`
- `block_color_gs`
- `block_color_rs`

### `vision_node`

- `min_area`
- `erode_iter`
- `dilate_iter`

### `tf_transformer_node`

- `target_interval`
- `cam_pos_x`
- `cam_pos_y`
- `cam_pos_z`

### `trajectory_node`

- `move_duration`
- `publish_rate`
- `test_mode`
- `approach_height`
- `lift_height`
- `home_height`
- `max_joint_velocity`

### `arm_driver_node`

- `use_serial`
- `serial_port`
- `baudrate`

## RViz

RViz 配置文件位于 [config/vision_grasp.rviz](./config/vision_grasp.rviz)。

`demo.launch.py` 会自动加载它，主要用于显示：

- 机械臂模型
- 仿真色块 marker
- 抓取目标 marker

## Upstream Asset Source

机械臂模型资产来自 AgileX Robotics 的 `agilex_open_class` 仓库：

- 仓库：<https://github.com/agilexrobotics/agilex_open_class>
- 子目录：`piper/piper_description`

当前包中复用了以下上游资产并做了本地适配：

- `description/meshes/*.STL`
- `description/piper.urdf`

本地适配主要包括：

- 将 mesh 路径改为 `package://vision_grasp/description/meshes/...`
- 将机械臂描述直接打包到 `vision_grasp` 中，便于单包 bringup
- 围绕该模型增加了视觉、坐标转换、IK 和抓取状态机节点

## Known Limitations

- 当前重点是仿真验证，不是完整量产级控制栈
- 检测结果和抓取目标当前仍使用 `PoseStamped` 承载内部语义，接口还可以继续工程化
- 真机串口协议和舵机映射仍是原型实现，部署前应单独校准和验证
