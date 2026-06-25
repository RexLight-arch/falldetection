# 基于昇腾边缘计算的人体跌倒检测与报警系统

面向独居老人、病房及安全监护场景，构建从模型训练、数据处理、指标评估到昇腾边缘部署的完整跌倒检测方案。系统支持图像、视频和摄像头实时推理，并通过 MQTT 上报跌倒事件、FFmpeg/RTMP 推送检测视频。

## 核心功能

- YOLO 跌倒检测模型训练、验证与测试
- 数据集格式转换和标签校验
- 图像、视频与摄像头实时检测
- 昇腾 OM 模型 NPU 推理
- MediaPipe 人体关键点与时序特征分析
- MQTT 跌倒告警与设备上线通知
- FFmpeg H.264 编码及 RTMP 视频推送
- PyQt5 训练、测试、推理和数据转换工具

## 技术亮点

1. **模型训练与改进**：基于 PyTorch 和 Ultralytics YOLOv8 训练跌倒检测模型，工程中保留 CBAM 模型配置与训练评估结果，用于增强关键姿态区域的特征表达。
2. **多特征跌倒判定**：融合人体检测框与 MediaPipe Pose 关键点，综合躯干高度比例、躯干—腿部角度、头部位置、垂直速度和身体倾斜角等特征。
3. **时序状态机降误报**：设计 `NORMAL → FALLING → FALLEN → RISING` 状态机，并结合滑动窗口投票、持续时间与恢复判断，避免单帧姿态造成误报。
4. **可追溯模型评估**：训练流程输出 Precision、Recall、mAP50、mAP50-95、混淆矩阵及 P/R/F1 曲线；现有 `train2` 记录中最高 mAP50 为 **0.958**，对应 mAP50-95 为 **0.654**。
5. **昇腾边缘推理**：使用 `ais_bench InferSession` 加载 OM 模型，在 Ascend NPU 上完成推理，并提供图像预处理、输出解析和检测框恢复流程。
6. **可视化工具链**：使用 PyQt5 集成训练、测试、推理、主题切换、终端日志和数据集转换，降低模型实验和演示门槛。
7. **视频与事件联动**：使用 FFmpeg 将检测画面编码为 H.264 并通过 RTMP 推送；跌倒事件通过 MQTT 上报，可联动移动端通知和历史记录。
8. **运行状态监控**：实时显示 FPS、检测状态和姿态结果，并使用日志和 `psutil` 记录异常及资源占用。

## 性能说明

- 当前公开脚本默认目标帧率为 **15 FPS**，实际帧率由昇腾型号、OM 模型、视频分辨率和预处理开销决定。
- RTMP 采用 `libx264` 的 `veryfast` 预设，以实时性优先。
- 端到端延迟还会受到摄像头采集、预处理、编码和网络状况影响，部署时应结合目标硬件单独压测。

## 检测流程

```text
摄像头 / 视频
      ↓
昇腾 NPU 人体检测
      ↓
MediaPipe 姿态关键点
      ↓
多特征投票 + 滑动窗口 + 状态机
      ├─ 正常：持续监测
      └─ 跌倒：MQTT 告警 + RTMP 视频推送
```

## 技术栈

`Python` · `PyTorch` · `Ultralytics YOLOv8` · `Ascend NPU` · `CANN` · `ais_bench` · `MediaPipe` · `OpenCV` · `PyQt5` · `MQTT` · `FFmpeg` · `RTMP`

## 主要文件

```text
.
├─ FALL_DOWN.py   # 跌倒检测、状态机、告警和推流主流程
├─ Third.py       # 昇腾 OM 模型推理版本
└─ 基于YOLOv8的跌倒检测(1)/
   └─ 基于YOLOv8的跌倒检测/
      ├─ youi/    # PyQt5 训练/测试/推理工具
      ├─ 测试/    # 测试样例
      ├─ runs/    # 训练指标与评估图表
      └─ main/    # 模型配置与测试入口
```

## 环境与运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install opencv-python mediapipe numpy paho-mqtt psutil
```

昇腾部署还需安装对应版本的 CANN、驱动与 `ais_bench`，准备转换后的 `.om` 模型，并修改模型路径、MQTT Broker 和 RTMP 地址。

```bash
python FALL_DOWN.py
```

## 模型与数据说明

模型权重、OM 模型、完整数据集、视频输出和压缩交付包未纳入 Git，以避免仓库膨胀和潜在的数据授权问题。仓库保留核心算法、可视化工具、部分测试样例及训练评估结果。
