# 机器狗冬令营

上海交大学生创新中心冬令营 · XGO-MINI（陆吾 Luwu Dynamics，15 自由度）课程资料、实验代码与体检记录。

[XGO 建图使用手册](docs/XGO建图使用手册.md)涵盖实时建图、保存地图、录包、离线回放与故障排查。

## 目录结构

```
.
├── src/
│   ├── robot/                     # 狗端采集、录包和控制
│   │   ├── webview.py             # 相机、雷达和 IMU 接口（:8000）
│   │   ├── lidar.py               # 共用雷达配置和体检入口
│   │   ├── record_bag.py          # 原始雷达和 IMU 录包
│   │   ├── dog_bridge.py          # 关节角度 HTTP 桥接（:8110）
│   │   ├── gait/                  # 狗端策略推理与 ONNX 模型
│   │   └── vision/                # 视觉抓取
│   ├── slam/                      # 实时建图与离线回放
│   ├── workstation/
│   │   ├── mapping/               # 相机、地图和网页遥控
│   │   └── teleop/                # 键盘遥控、Mac 侧策略推理
│   └── diagnostics/               # IMU 体检
├── data/
│   ├── bags/                      # 建图原始录包
│   └── diagnostics/               # 雷达与 IMU 体检原始数据
└── docs/
    ├── archives/                  # 离线网页存档
    ├── course_materials/          # 课件与教材
    ├── reports/                   # 体检报告和建图对照图
    └── screenshots/               # 运行截图
```

## 当前设备

| 项 | 值 |
|---|---|
| 地址 | `172.18.12.148`（WiFi 的 DHCP，可能变；以狗屏幕显示的 IP 为准） |
| 主机名 | `raspberrypi` |
| 登录 | `ssh luwu@172.18.12.148`（用户主目录为 `/home/pi`） |
| 固件 | `M-7.0.0b8` |
| 环境 | Python 3.13，`xgolib` 已装（`/usr/local/lib/python3.13/dist-packages`） |
| 雷达 | YDLidar Tmini Plus，`/dev/ttyUSB0 @ 230400` |
| IMU | `/dev/ttyAMA0 @ 115200`（xgolib 直接读） |

## 部署建图脚本

```bash
ssh luwu@172.18.12.148 'mkdir -p /home/pi/tools'
scp src/robot/webview.py src/robot/lidar.py luwu@172.18.12.148:/home/pi/
scp src/slam/slam2d.py src/robot/record_bag.py luwu@172.18.12.148:/home/pi/tools/
```

复制完成后按[建图使用手册](docs/XGO建图使用手册.md#2-从头启动或重开一张图)启动。

## 建图链路的设计

- 狗端 `webview.py` 每次成功读取一圈就发布一帧。旧版按 100 ms 分箱，会把接近阈值的相邻两圈并到一帧。重启前应等原采集进程释放 `/dev/ttyUSB0`，两个进程同时读取会造成串口校验错误。
- 建图使用狗端时间戳和雷达似然场估计位置及航向。IMU 航向先按度计算最短角差，再转成弧度作为搜索初值。roll 或 pitch 超过 25° 时跳过该帧，匹配分过低或出现异常跳变时保留上一可信位姿。
- 地图分辨率为 5 cm。每帧中同一栅格只更新一次，射线命中端点标为占据，沿途的格子标为空闲。
- `--record FILE` 录制参与建图的帧，完整雷达和 IMU 流用 `src/robot/record_bag.py` 录制。实时端按帧号消费，链路长时间积压时可触发丢帧。

### 手推录包复测

标准输入是 [手推首圈录包](data/bags/2026-11-20_1302_手推首圈.jsonl)。用狗端原配置和修复版回放同一份 1,280 帧数据，地图与轨迹见[建图修复对比](docs/reports/建图修复对比.png)。

```bash
python3 src/slam/slam2d.py --replay data/bags/2026-11-20_1302_手推首圈.jsonl \
  --no-view --map-out /tmp/xgo-map.png
```

修复版在狗端处理 1,265 帧，因倾角跳过 15 帧，匹配拒绝 0 帧。这次验证使用手推轨迹与静止实时流，步态行走时的误差仍需另行采集。若要处理连续行走引入的扫描畸变，需要先记录束级时间并校准雷达与 IMU。更长轨迹的累计漂移可再用子图回环和位姿图优化处理。

本次在狗端通过后台进程提供 [实时建图页面](http://172.18.12.148:8010)。本次启动的 PID 和日志位于 `/home/pi/.local/state/xgo-slam/`。重新启动前可用 `ps` 核对 PID 后发 `SIGTERM`，并用 `fuser /dev/ttyUSB0` 确认雷达已释放。

## 坐标约定（重要）

- 雷达原始角度是**顺时针正方向**（实测：椅子放狗左侧 1 m，原始角 −66°）。`webview.py` 在发布前已统一翻成**逆时针**（左 = 正，与 ROS 一致），所以 `slam2d.py` 里 IMU 航向符号取 **+1**。
- 因此 **2026-11-20 12:46 之前录的包是旧（顺时针）约定**，回放时要先镜像角度，否则地图会左右反。
- 已验证（2026-11-20 13:00）：椅子放狗左侧 1 m，修前原始角 **−66°**，翻转后为 **+52°~+88°（重心 +72°）**，即左 = 正，坐标约定与真实世界一致。

## 已知问题

- `dog.read_imu()` 读到的是死数据（固件 0x65 寄存器没实现，M-6.0.2 / M-7.0.0b8 都有），角度请用 `read_roll/pitch/yaw`。
- `yaw` 静止时不更新（固件死区），只能做粗转向，别用于精细角度。
- 各脚本里的默认 IP 仍是旧地址 `172.18.11.103`，换机时用 `--host` 指定。
- **WiFi 省电**会让 ping 从 8 ms 飙到 2000 ms，直接毁掉实时建图。已在狗上关闭并做成开机自启（`wifi-powersave-off.service`）。

最近一次体检见[机器狗硬件体检报告](docs/reports/机器狗硬件体检-2026-11-20.md)。
