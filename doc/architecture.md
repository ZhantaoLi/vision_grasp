# vision_grasp Architecture

## Scope

`vision_grasp` 是一个单包 ROS 2 Jazzy 视觉抓取原型，当前目标是稳定支撑仿真链路：

1. 生成相机图像和相机内参
2. 检测彩色目标
3. 将像素目标映射到机械臂基座坐标系
4. 规划抓取轨迹并发布关节状态
5. 在仿真模式或真机模式下消费关节命令

当前工程重点是：

- 仿真可运行
- 包级依赖和 launch 接线已经收口
- 已有基础自动化测试

不在当前范围内的内容：

- 重构为多包工作区
- 重写 URDF
- 更换夹爪几何或夹爪闭合符号逻辑
- 将内部 `PoseStamped` 协议替换为自定义消息

## Directory Structure

```text
src/vision_grasp/
├── CHANGELOG.rst
├── LICENSE
├── README.md
├── config/
│   ├── params.yaml
│   └── vision_grasp.rviz
├── description/
│   ├── meshes/
│   │   ├── base_link.STL
│   │   ├── link1.STL
│   │   ├── link2.STL
│   │   ├── link3.STL
│   │   ├── link4.STL
│   │   ├── link5.STL
│   │   ├── link6.STL
│   │   ├── link7.STL
│   │   └── link8.STL
│   └── piper.urdf
├── doc/
│   ├── architecture.md
│   └── architecture_diagram.html
├── launch/
│   ├── demo.launch.py
│   └── pipeline.launch.py
├── package.xml
├── resource/vision_grasp
├── setup.cfg
├── setup.py
├── test/
│   ├── test_ik_utils.py
│   ├── test_metadata_and_launch.py
│   ├── test_pipeline_launch.py
│   └── test_tf_transformer_node.py
└── vision_grasp/
    ├── __init__.py
    ├── arm_driver_node.py
    ├── camera_node.py
    ├── ik_utils.py
    ├── trajectory_support.py
    ├── tf_transformer_node.py
    ├── trajectory_node.py
    └── vision_node.py
```

## Runtime Topology

### Launch Entry Points

- `pipeline.launch.py`
  - 启动抓取管线
  - 不启动 RViz
- `demo.launch.py`
  - 启动相同管线
  - 额外启动 RViz2 并加载 `vision_grasp.rviz`

两个 launch 文件现在都从同一个 `config/params.yaml` 传参给：

- `camera_node`
- `vision_node`
- `tf_transformer_node`
- `trajectory_node`
- `arm_driver_node`

## Node Responsibilities

### `camera_node`

输入：

- 无

输出：

- `/image_raw`
- `/camera_info`
- `/block_markers`

职责：

- 在仿真模式下生成灰底图像和 4 个彩色方块
- 在真机模式下读取 OpenCV 摄像头
- 发布相机内参和 RViz marker

### `vision_node`

输入：

- `/image_raw`

输出：

- `/detected_objects`
- `/debug_image`

职责：

- 使用 HSV 阈值检测红、绿、蓝、黄方块
- 计算检测框中心和尺寸
- 输出调试图像

### `tf_transformer_node`

输入：

- `/camera_info`
- `/detected_objects`

输出：

- `/grasp_target`
- `base_link -> camera_link` 静态 TF

职责：

- 维护相机内参
- 使用固定相机外参
- 通过射线和平面求交把像素目标映射到 `base_link`

### `trajectory_node`

输入：

- `/grasp_target`

输出：

- `/joint_states`
- `/grasp_marker`

职责：

- 读取 `piper.urdf`
- 构建 `base_link -> link6` 运动链
- 计算 IK、接近点、抓取点、抬升点和归位点
- 执行抓取状态机
- 发布插值后的关节状态

### `arm_driver_node`

输入：

- `/joint_states`

输出：

- 仿真模式下无硬件输出
- 真机模式下串口舵机命令

职责：

- 在仿真模式下作为硬件消费端占位
- 在真机模式下将 joint1-6 角度下发到舵机

### `ik_utils.py`

职责：

- 解析 URDF 关节链
- 计算 FK
- 计算多关节链末端位置 IK
- 提供碰地检查需要的链路 z 坐标计算

### `trajectory_support.py`

职责：

- 保存抓取状态常量
- 保存夹爪打开/安全闭合常量
- 提供状态迁移、线性插值、运动时长计算、步长限制和日志格式化等纯函数

## Data Flow

```text
camera_node
  ├─ /image_raw -----------------------> vision_node
  ├─ /camera_info ---------------------> tf_transformer_node
  └─ /block_markers -------------------> RViz

vision_node
  └─ /detected_objects ---------------> tf_transformer_node

tf_transformer_node
  ├─ static TF base_link→camera_link -> TF tree
  └─ /grasp_target -------------------> trajectory_node

trajectory_node
  ├─ /joint_states -------------------> arm_driver_node
  ├─ /joint_states -------------------> robot_state_publisher
  └─ /grasp_marker -------------------> RViz
```

## Topic Contract

| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | `camera_node` | `vision_node` |
| `/camera_info` | `sensor_msgs/CameraInfo` | `camera_node` | `tf_transformer_node` |
| `/block_markers` | `visualization_msgs/MarkerArray` | `camera_node` | RViz |
| `/detected_objects` | `geometry_msgs/PoseStamped` | `vision_node` | `tf_transformer_node` |
| `/debug_image` | `sensor_msgs/Image` | `vision_node` | 调试侧 |
| `/grasp_target` | `geometry_msgs/PoseStamped` | `tf_transformer_node` | `trajectory_node` |
| `/joint_states` | `sensor_msgs/JointState` | `trajectory_node` | `arm_driver_node`, `robot_state_publisher` |
| `/grasp_marker` | `visualization_msgs/Marker` | `trajectory_node` | RViz |

## Internal Encodings

当前链路中有两个内部协议是通过 `PoseStamped` 复用实现的。

### `/detected_objects`

- `header.frame_id`
  - 物体类型名，例如 `red_block`
- `pose.position.x`
  - 像素中心 `u`
- `pose.position.y`
  - 像素中心 `v`
- `pose.position.z`
  - 像素框宽度
- `pose.orientation.x`
  - 像素框高度
- `pose.orientation.w`
  - 简化置信度

### `/grasp_target`

- `header.frame_id`
  - 固定为 `base_link`
- `pose.position.x/y/z`
  - 抓取目标在基座系中的位置
- `pose.orientation.x`
  - 色块物理宽度，供夹爪闭合使用
- `pose.orientation.w`
  - 当前固定为 `1.0`

这套协议当前工作正常，但从工程化角度看仍然属于“原型内部约定”，后续若继续收口，可以迁移到自定义 `msg`。

## Mechanical Model

### URDF Source

- 机械臂模型来自 AgileX `agilex_open_class` 的 `piper/piper_description`
- 本地使用的文件是 `description/piper.urdf`
- 当前只读取和最小适配已有 URDF，不重新生成

### Kinematic Chain

主运动链：

```text
base_link
  -> link1
  -> link2
  -> link3
  -> link4
  -> link5
  -> link6
```

夹爪分支：

```text
link6 -> joint7 -> link7
link6 -> joint8 -> link8
```

### Gripper Geometry Constraints

当前项目内已经确认的约束：

- `joint7` 使用负值闭合
- `joint8` 使用正值闭合
- 闭合公式：

```text
joint7 = -W / 2
joint8 = +W / 2
```

其中 `W` 是物体宽度。

这属于已验证约束，后续工程化修改不应改变其符号定义。

## Parameters

统一参数文件：`config/params.yaml`

### `camera_node`

- `use_camera`
- `camera_id`
- `fps`
- `width`
- `height`
- `cam_pos_x`
- `cam_pos_y`
- `cam_pos_z`

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
- `urdf_file`
- `base_link`
- `tip_link`
- `approach_height`
- `lift_height`
- `home_height`
- `max_joint_velocity`

### `arm_driver_node`

- `use_serial`
- `serial_port`
- `baudrate`

## Test Coverage

当前包内已有以下自动化测试：

- `test_metadata_and_launch.py`
  - 检查交付文件存在
  - 检查 `package.xml` 依赖声明
  - 检查 launch 文件是否给 `tf_transformer_node` 正确传参
- `test_ik_utils.py`
  - 检查夹爪关节限位
  - 检查运动链提取
  - 检查简化两关节 IK 的可达性
- `test_tf_transformer_node.py`
  - 检查中心像素投影结果
  - 检查 CameraInfo 对内参更新
- `test_trajectory_support.py`
  - 检查夹爪闭合符号
  - 检查状态机迁移
  - 检查插值、步长限制和时长计算
- `test_pipeline_launch.py`
  - smoke test
  - 验证整条仿真管线关键节点能够拉起

## Current Engineering Boundaries

已经完成的收口：

- 包依赖声明已补齐
- launch 参数链已闭合
- README / LICENSE / CHANGELOG 已补齐
- 包级单测和 smoke test 已建立

仍然保留为原型实现的部分：

- `PoseStamped` 承载内部业务协议
- `trajectory_node.py` 仍然承担较多职责
- 真机串口驱动仍是硬编码映射
- 尚未拆成多包架构

## Recommended Next Refactor

如果继续工程化，建议顺序如下：

1. 只拆 `trajectory_node.py` 的内部模块，不改行为
2. 再引入显式接口消息，替代 `PoseStamped` 复用
3. 最后再考虑拆成 `interfaces / perception / planning / hardware / bringup`

这个顺序风险最低，也最符合当前“仿真已通，逐步收口”的状态。
