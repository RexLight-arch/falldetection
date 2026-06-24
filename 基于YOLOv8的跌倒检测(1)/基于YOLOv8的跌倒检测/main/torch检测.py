import torch

# 检查版本
print(torch.__version__)  # 输出版本，如 2.1.0

# 检查 GPU 支持（NVIDIA）
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    # 测试 GPU 计算
    x = torch.tensor([1.0]).cuda()
    print(x)  # 输出 tensor([1.0], device='cuda:0')
else:
    print("未检测到 GPU，使用 CPU 版本")

# Apple Silicon 专用检查
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS 加速已启用")
    y = torch.tensor([1.0], device="mps")
    print(y)
import os

print("当前项目路径为：", os.getcwd())

import os
print("当前工作目录:", os.getcwd())


import os

ultralytics_config_path = os.path.expandvars(r"%APPDATA%\Ultralytics\settings.json")

if os.path.exists(ultralytics_config_path):
    os.remove(ultralytics_config_path)
    print("Ultralytics 缓存配置已清除 ✅")
else:
    print("未找到 Ultralytics 缓存配置文件")


# yolo detect train data=datasets/expression/loopy.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=200 batch=4 lr0=0.01 resume=True
# yolo detect train data=datasets/coco128/coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=200 batch=4 lr0=0.01 resume=True


# yolo detect train data=datasets/expression/loopy.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 batch=16 lr0=0.001