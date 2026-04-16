# NGSIM-US-101 跟驰行为数据处理工具

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

本项目提供了一套完整的NGSIM-US-101数据集处理流程，将原始轨迹数据转换为适用于**车辆跟驰（Car-following）任务**的机器学习训练数据。

## 目录

1. [项目概述](#1-项目概述)
2. [数据处理方法](#2-数据处理方法)
3. [使用方法](#3-使用方法)
4. [数据格式说明](#4-数据格式说明)
5. [输出文件结构](#5-输出文件结构)
6. [项目结构](#6-项目结构)
7. [引用](#7-引用)

---

## 1. 项目概述

NGSIM-US-101数据集记录了美国US-101高速公路上的车辆轨迹数据，包含三个时间段（7:50-8:05, 8:05-8:20, 8:20-8:35）的高精度车辆运动信息。

本项目通过8个步骤的处理流程，将原始轨迹数据转换为标准化的时序训练数据，可直接用于深度学习模型（如LSTM、Transformer等）的训练。

### 核心特性

- **完整的处理流水线**：从原始数据到训练集的一站式处理
- **Savitzky-Golay平滑**：有效消除定位噪声，确保物理合理性
- **严格的跟驰对筛选**：基于时空连续性，剔除换道行为
- **多版本输出**：提供10秒和20秒两种时间窗口的训练数据
- **模型验证**：内置LSTM基准模型验证数据可用性

---

## 2. 数据处理方法

本项目的数据处理流程包含以下主要步骤：

(1) **轨迹平滑与去噪**：使用Savitzky-Golay滤波器消除原始数据中的定位抖动
(2) **坐标转换**：将数据从英制单位转换为公制单位，并进行异常值清洗
(3) **跟驰对提取**：基于Vehicle_ID和Preceeding字段匹配前后车，并筛选时空连续性符合要求的配对
(4) **特征计算**：计算相对距离、相对速度和后车速度作为输入特征，后车加速度作为预测标签
(5) **数据分段与标准化**：使用滑动窗口分割数据，进行标准化处理后导出为HDF5格式
(6) **模型验证**：使用LSTM模型验证数据集的可用性

---

## 3. 使用方法

### 1. 环境准备

```bash
# 创建conda环境
conda create -n py-310 python=3.10
conda activate py-310

# 安装依赖
pip install -r requirement.txt
```

### 2. 数据准备

将NGSIM-US-101原始数据放置在 `./US-101/` 目录下：
```
US-101/
├── 0750am-0805am/
│   └── trajectories-0750am-0805am.txt
├── 0805am-0820am/
│   └── trajectories-0805am-0820am.txt
└── 0820am-0835am/
    └── trajectories-0820am-0835am.txt
```

### 3. 运行处理流程

```bash
# 步骤1: 轨迹平滑与去噪
python code/step1_denoising.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# 步骤2: 坐标转换与清洗
python code/step2_coordinate_conversion.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# 步骤3: 跟驰对提取
python code/step3_car_following.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# 步骤4: 特征计算
python code/step4_feature_engineering.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# 步骤5&6: 数据分段与HDF5导出
python code/step5_segmentation.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

# 步骤8: LSTM模型验证
python code/step8_lstm_validation.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc
```

---

## 4. 数据格式说明

(1) 原始数据字段（NGSIM-US-101）

| 字段名 | 说明 | 单位 |
|--------|------|------|
| Vehicle_ID | 车辆唯一标识符 | - |
| Frame_ID | 帧编号 | - |
| Global_Time | 全局时间戳 | 毫秒 |
| Local_X | 本地横向坐标 | feet |
| Local_Y | 本地纵向坐标 | feet |
| v_Length | 车辆长度 | feet |
| v_Width | 车辆宽度 | feet |
| v_Class | 车辆类型（1=摩托车，2=轿车，3=卡车） | - |
| v_Vel | 车辆速度 | mph |
| v_Acc | 车辆加速度 | ft/s² |
| Lane_ID | 车道编号 | - |
| Preceeding | 前车ID（0表示无前车） | - |
| Space_Headway | 空间车头间距 | feet |
| Time_Headway | 时间车头间距 | s |

(2) 处理后数据格式

**HDF5格式** (`train_10s.h5` / `train_20s.h5`)：

```python
import h5py

with h5py.File('code/output/train_10s.h5', 'r') as f:
    X = f['X'][:]  # shape: (n_samples, 100, 3)
    y = f['y'][:]  # shape: (n_samples, 100)

    print(f"Samples: {f.attrs['n_samples']}")
    print(f"Sequence length: {f.attrs['seq_len']}")
    print(f"Features: {f.attrs['n_features']}")
```

(3) 特征维度说明

- **X** (输入特征): `[spacing, v_rel, v_follower]`
  - `spacing`: 前后车车身净间距 (m)
  - `v_rel`: 前车速度 - 后车速度 (m/s)，正值表示后车正在追赶
  - `v_follower`: 后车速度 (m/s)

- **y** (标签): 后车加速度 `acc_follower` (m/s²)

---

## 5. 输出文件结构

```
.
├── code/
│   ├── output/                    # 处理后的数据文件
│   │   ├── trajectories_smoothed_{period}.csv
│   │   ├── trajectories_cleaned_{period}.csv
│   │   ├── car_following_pairs_{period}.csv
│   │   ├── features_{period}.csv
│   │   ├── train_10s_{period}.h5
│   │   ├── train_20s_{period}.h5
│   │   └── lstm_model_{period}.pth
│   ├── step1_denoising.py         # 步骤1: 轨迹平滑
│   ├── step2_coordinate_conversion.py  # 步骤2: 坐标转换
│   ├── step3_car_following.py     # 步骤3: 跟驰对提取
│   ├── step4_feature_engineering.py    # 步骤4: 特征计算
│   ├── step5_segmentation.py      # 步骤5&6: 分段与持久化
│   └── step8_lstm_validation.py   # 步骤8: 模型验证
├── doc/                           # 分析报告和可视化
│   ├── pic/                       # 生成的图像
│   │   ├── denoising_vehicle_*.png
│   │   ├── coordinate_conversion.png
│   │   ├── car_following_pairs.png
│   │   ├── feature_distribution.png
│   │   ├── segmentation.png
│   │   └── lstm_validation.png
│   ├── denoising_report.md
│   ├── coordinate_conversion_report.md
│   ├── car_following_report.md
│   ├── feature_report.md
│   ├── segmentation_report.md
│   └── model_validation_report.md
└── US-101/                        # 原始数据（不上传到Git）
    ├── 0750am-0805am/
    ├── 0805am-0820am/
    └── 0820am-0835am/
```

---

## 6. 项目结构

```
.
├── .gitignore             			# Git忽略文件配置
├── README.md              		# 英文版说明文档
├── README_CN.md           		# 中文版说明文档
├── Requirements_document_new.md  	# 详细需求文档
├── requirement.txt         		# Python依赖包列表
├── code/                  		# 代码目录
│   ├── *.py               			# 处理脚本
│   └── output/            		# 输出数据
├── doc/                    		# 文档和可视化
│   ├── *.md               			# 分析报告
│   └── pic/               			# 图像
└── US-101/                 		# 原始数据
```

---

## 7. 引用

如果您在研究中使用了本项目处理的数据，请引用：

1. NGSIM数据集原始文献
2. 本项目GitHub仓库

---

## License

MIT License

---

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。