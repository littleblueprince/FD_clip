# -*- coding: utf-8 -*-
# @Time    : 2023/11/14 10:38
# @Author  : blue
# @Description :


import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"发现 {gpu_count} 个 CUDA 设备")

    # 获取当前使用的CUDA设备索引
    current_device = torch.cuda.current_device()
    print(f"当前使用的 CUDA 设备索引: {current_device}")

    # 获取当前CUDA设备的名称
    current_device_name = torch.cuda.get_device_name(current_device)
    print(f"当前CUDA设备的名称: {current_device_name}")

    # 打印CUDA设备的性能信息
    for i in range(gpu_count):
        gpu = torch.cuda.get_device_properties(i)
        print(f"CUDA 设备 {i}: {gpu.name}, 计算能力: {gpu.major}.{gpu.minor}")
else:
    print("未发现可用的 CUDA 设备")

