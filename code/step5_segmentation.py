"""
步骤5 & 6: 数据标准化与分段 + 持久化
- 滑动窗口切片：10秒和20秒两个版本
- 保存为HDF5格式

Usage:
    # 处理单个时间段
    python code/step5_segmentation.py --period 0750am-0805am --data-dir US-101

    # 处理所有时间段
    python code/step5_segmentation.py --data-dir US-101

    # 指定输出和文档目录
    python code/step5_segmentation.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

Arguments:
    --period      时间段 (可选, 如 "0750am-0805am", "0805am-0820am", "0820am-0835am")
    --data-dir    输入数据目录名称 (默认: US-101)，用于构建输入/输出路径
    --output-dir  输出根目录 (默认: code/output)，数据将保存到 output-dir/data-dir-name/period
    --doc-dir     文档根目录 (默认: doc)，报告和图片将保存到 doc-dir/data-dir-name/period
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置（将在main中设置）
INPUT_FILE = None
OUTPUT_DIR = 'code/output'
DOC_DIR = 'doc'
PIC_DIR = 'doc/pic'

# 分段参数
WINDOW_SIZES = [10, 20]  # 秒
OVERLAP = 0.5  # 50%重叠
FRAME_RATE = 10  # 10fps (NGSIM数据每0.1秒一帧)

def load_features():
    """加载特征数据"""
    print("Loading features data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records")
    return df

def create_sequences(df, window_size_sec, overlap=0.5):
    """使用滑动窗口创建时间序列片段"""
    window_size_frames = int(window_size_sec * FRAME_RATE)
    step_size = int(window_size_frames * (1 - overlap))

    print(f"Creating sequences: window={window_size_sec}s ({window_size_frames} frames), step={step_size} frames")

    sequences = []
    labels = []
    metadata = []

    # 按车辆分组
    grouped = df.groupby('Vehicle_ID')

    for vehicle_id, group in grouped:
        group = group.sort_values('Global_Time').reset_index(drop=True)

        # 检查是否有前车信息
        if 'leader_y' not in group.columns or group['leader_y'] is None:
            continue

        n_frames = len(group)

        # 滑动窗口
        start = 0
        while start + window_size_frames <= n_frames:
            end = start + window_size_frames

            # 提取窗口数据
            window = group.iloc[start:end]

            # 检查数据完整性
            if window['spacing'].notna().all() and \
               window['v_rel'].notna().all() and \
               window['v_follower'].notna().all() and \
               window['acc_follower'].notna().all():

                # 特征: [spacing, v_rel, v_follower]
                x = window[['spacing', 'v_rel', 'v_follower']].values

                # 标签: 加速度 (每个时刻)
                y = window['acc_follower'].values

                sequences.append(x)
                labels.append(y)
                metadata.append({
                    'vehicle_id': vehicle_id,
                    'start_time': window['Global_Time'].iloc[0],
                    'end_time': window['Global_Time'].iloc[-1],
                    'n_points': len(window)
                })

            start += step_size

    sequences = np.array(sequences)
    labels = np.array(labels)

    print(f"Created {len(sequences)} sequences, shape: {sequences.shape}")

    return sequences, labels, metadata

def normalize_features(sequences):
    """标准化特征"""
    print("Normalizing features...")

    # 重塑以便标准化
    n_seq, seq_len, n_features = sequences.shape
    sequences_flat = sequences.reshape(-1, n_features)

    # 标准化
    scaler = StandardScaler()
    sequences_normalized = scaler.fit_transform(sequences_flat)
    sequences_normalized = sequences_normalized.reshape(n_seq, seq_len, n_features)

    print(f"Normalized shape: {sequences_normalized.shape}")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature stds: {scaler.scale_}")

    return sequences_normalized, scaler

def save_to_hdf5(sequences, labels, filename):
    """保存到HDF5文件"""
    print(f"Saving to {filename}...")

    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=sequences, compression='gzip')
        f.create_dataset('y', data=labels, compression='gzip')
        f.attrs['n_samples'] = len(sequences)
        f.attrs['seq_len'] = sequences.shape[1]
        f.attrs['n_features'] = sequences.shape[2]

    print(f"Saved {len(sequences)} samples to {filename}")

def visualize_segmentation(sequences, window_sizes):
    """可视化分段结果"""
    print("Generating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 样本数量对比
    ax1 = axes[0]
    window_labels = [f'{ws}s' for ws in window_sizes]
    sample_counts = [len(s) for s in sequences]
    bars = ax1.bar(window_labels, sample_counts, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Number of Samples by Window Size')
    for bar, count in zip(bars, sample_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    ax1.grid(True, alpha=0.3)

    # 2. 序列长度分布
    ax2 = axes[1]
    seq_lens = [s.shape[1] for s in sequences]
    ax2.hist(seq_lens, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Count')
    ax2.set_title('Sequence Length Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'segmentation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {PIC_DIR}/segmentation.png")

def generate_segmentation_report(results, window_sizes):
    """生成分段报告"""

    report = f"""# 数据标准化与分段分析报告

## 1. 概述

本报告描述了数据的标准化处理和滑动窗口分段时间。

## 2. 处理方法

### 2.1 滑动窗口参数

| 参数 | 值 |
|------|------|
| 帧率 | {FRAME_RATE} fps |
| 重叠比例 | {OVERLAP*100:.0f}% |

### 2.2 窗口大小

生成了两个版本的序列数据：
- 10秒版本：{results[0][0]} 个样本
- 20秒版本：{results[1][0]} 个样本

### 2.3 特征标准化

使用StandardScaler进行标准化处理：
- spacing (相对距离)
- v_rel (相对速度)
- v_follower (后车速度)

## 3. 数据统计

| 版本 | 样本数 | 序列长度 | 特征维度 |
|------|--------|----------|----------|
"""

    for i, ws in enumerate(window_sizes):
        n_samples, seq_len, n_features = results[i]
        report += f"| {ws}s | {n_samples:,} | {seq_len} | {n_features} |\n"

    report += f"""
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
"""

    report_path = os.path.join(DOC_DIR, 'segmentation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    return report

def main():
    """主函数"""
    global INPUT_FILE, OUTPUT_DIR, DOC_DIR, PIC_DIR

    parser = argparse.ArgumentParser(description='Segmentation and HDF5 Export')
    parser.add_argument('--period', type=str, default=None,
                       help='Time period to process (e.g., "0750am-0805am")')
    parser.add_argument('--data-dir', type=str, default='US-101',
                       help='Input data directory name')
    parser.add_argument('--output-dir', type=str, default='code/output',
                       help='Output directory for data')
    parser.add_argument('--doc-dir', type=str, default='doc',
                       help='Output directory for docs')
    args = parser.parse_args()

    period = args.period
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))

    # 构建输出路径: output-dir/data-dir/period
    # 构建文档路径: doc-dir/data-dir/period
    if period:
        OUTPUT_DIR = os.path.join(args.output_dir, data_dir_name, period)
        DOC_DIR = os.path.join(args.doc_dir, data_dir_name, period)
        INPUT_FILE = os.path.join(args.output_dir, data_dir_name, period, f'features_{period}.csv')
    else:
        OUTPUT_DIR = os.path.join(args.output_dir, data_dir_name)
        DOC_DIR = os.path.join(args.doc_dir, data_dir_name)
        INPUT_FILE = os.path.join(args.output_dir, data_dir_name, 'features.csv')

    PIC_DIR = os.path.join(DOC_DIR, 'pic')

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PIC_DIR, exist_ok=True)

    title = "Step 5 & 6: Segmentation and HDF5 Export"
    if period:
        title += f" - {period}"
    print("=" * 60)
    print(title)
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] Loading features data...")
    df = load_features()

    results = []
    all_sequences = []  # 保存所有窗口大小的序列

    # 2. 对每个窗口大小进行处理
    for window_size in WINDOW_SIZES:
        print(f"\n--- Processing {window_size}s window ---")

        # 创建序列
        sequences, labels, metadata = create_sequences(df, window_size, OVERLAP)

        # 保存序列用于可视化
        all_sequences.append(sequences)

        # 标准化
        sequences_normalized, scaler = normalize_features(sequences)

        # 保存到HDF5
        if period:
            filename = os.path.join(OUTPUT_DIR, f'train_{window_size}s_{period}.h5')
        else:
            filename = os.path.join(OUTPUT_DIR, f'train_{window_size}s.h5')
        save_to_hdf5(sequences_normalized, labels, filename)

        results.append((len(sequences), sequences.shape[1], sequences.shape[2]))

    # 3. 可视化
    print("\n[3/4] Generating visualizations...")
    visualize_segmentation(all_sequences, WINDOW_SIZES)

    # 4. 生成报告
    print("\n[4/4] Generating report...")
    generate_segmentation_report(results, WINDOW_SIZES)

    print("\n" + "=" * 60)
    print("Steps 5 & 6 Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()