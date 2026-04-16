# 数据标准化与分段分析报告

## 1. 概述

本报告描述了数据的标准化处理和滑动窗口分段时间。

## 2. 处理方法

### 2.1 滑动窗口参数

| 参数 | 值 |
|------|------|
| 帧率 | 10 fps |
| 重叠比例 | 50% |

### 2.2 窗口大小

生成了两个版本的序列数据：
- 10秒版本：7761 个样本
- 20秒版本：3084 个样本

### 2.3 特征标准化

使用StandardScaler进行标准化处理：
- spacing (相对距离)
- v_rel (相对速度)
- v_follower (后车速度)

## 3. 数据统计

| 版本 | 样本数 | 序列长度 | 特征维度 |
|------|--------|----------|----------|
| 10s | 7,761 | 100 | 3 |
| 20s | 3,084 | 200 | 3 |

## 4. 输出文件

- 10秒版本: `code/output/train_10s.h5`
- 20秒版本: `code/output/train_20s.h5`
- 可视化图像: `doc/pic/segmentation.png`
- 分析报告: `doc/segmentation_report.md`

## 5. HDF5文件结构

```
train_10s.h5 / train_20s.h5
├── X (n_samples, seq_len, n_features)  # 输入特征
├── y (n_samples, seq_len)              # 标签（加速度）
└── 属性
    ├── n_samples
    ├── seq_len
    └── n_features
```
