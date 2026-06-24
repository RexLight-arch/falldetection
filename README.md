# 基于昇腾芯片的边缘侧跌倒检测与报警系统

面向养老看护与公共安全场景的边缘 AI 跌倒检测系统。项目在昇腾边缘设备上完成视频采集、人体检测、姿态关键点分析与跌倒状态判断，并通过 MQTT 上报告警、通过 FFmpeg 推送实时视频。

## 项目亮点

- 使用昇腾 `ais_bench InferSession` 加载 OM 模型，在 NPU 侧完成人体目标检测。
- 融合 MediaPipe Pose 关键点与人体框特征，提高单一姿态判断的鲁棒性。
- 使用高度比例、躯干角度、头部位置、垂直速度等多特征投票判断跌倒。
- 设计 `NORMAL → FALLING → FALLEN → RISING` 状态机，降低瞬时误报。
- 通过 MQTT 发布设备在线状态和跌倒告警。
- 使用 FFmpeg/RTMP 推送叠加检测结果的实时视频，并记录运行日志与资源占用。

## 检测流程

```text
摄像头输入
   ↓
昇腾 NPU 人体检测
   ↓
MediaPipe 姿态关键点
   ↓
时序特征 + 多指标投票 + 状态机
   ├─ 正常：继续监测
   └─ 跌倒：MQTT 告警 + RTMP 视频推送
```

## 技术栈

`Python` · `Ascend NPU` · `CANN` · `ais_bench` · `YOLO` · `MediaPipe` · `OpenCV` · `MQTT` · `FFmpeg` · `RTMP` · `PyQt5`

## 主要文件

```text
.
├─ FALL_DOWN.py   # 完整跌倒检测、状态机、告警和推流流程
├─ Third.py       # 昇腾 OM 模型推理版本
└─ 基于YOLOv8的跌倒检测(1)/
   └─ 基于YOLOv8的跌倒检测/
      ├─ youi/    # PyQt5 训练/测试可视化工具
      ├─ 测试/    # 测试样例
      └─ runs/    # 部分训练评估结果
```

## 环境与运行

推荐在已安装 CANN 与 Ascend 驱动的昇腾设备上运行。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install opencv-python mediapipe numpy paho-mqtt psutil
```

昇腾版本还需按设备环境安装 `ais_bench`，准备转换后的 `.om` 模型，并修改 `Third.py` 中的模型路径。运行前还需按部署环境配置 MQTT Broker、告警主题和 RTMP 地址。

```bash
python FALL_DOWN.py
```

## 模型与数据说明

模型权重、OM 模型、完整数据集、视频输出和压缩交付包未纳入 Git 版本管理，以避免仓库膨胀及潜在的数据授权问题。仓库保留核心算法代码、UI 工具和部分评估结果用于项目展示。

