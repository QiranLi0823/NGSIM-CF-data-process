"""
步骤3: 跟驰对 (Car-following Pairs) 提取
- 唯一ID匹配：根据Vehicle_ID和Preceeding提取前后车配对
- 时空连续性检查：同一车道，共存时间超过15秒
- 剔除换道行为

Usage:
    # 处理单个时间段
    python code/step3_car_following.py --period 0750am-0805am

    # 处理所有时间段
    python code/step3_car_following.py

    # 指定输出和文档目录
    python code/step3_car_following.py --period 0750am-0805am --output-dir code/output --doc-dir doc

Arguments:
    --period      时间段 (可选, 如 "0750am-0805am", "0805am-0820am", "0820am-0835am")
    --output-dir  输出目录 (默认: code/output)
    --doc-dir     文档目录 (默认: doc)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量（将在main中设置）
INPUT_FILE = 'code/output/trajectories_cleaned.csv'
OUTPUT_DIR = 'code/output'
DOC_DIR = 'doc'
PIC_DIR = 'doc/pic'

# 跟驰参数
MIN_DURATION = 15  # 最小共存时间（秒）
SAME_LANE = True  # 是否必须在同一车道

def load_cleaned_data():
    """加载清洗后的数据"""
    print("Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} records")
    return df

def detect_lane_changes(df):
    """检测换道行为"""
    print("Detecting lane changes...")

    # 按车辆和时间排序
    df = df.sort_values(['Vehicle_ID', 'Global_Time']).reset_index(drop=True)

    # 检测车道变化
    df['lane_change'] = df.groupby('Vehicle_ID')['Lane_ID'].diff().abs() > 0

    # 标记换道车辆
    lane_changing_vehicles = df[df['lane_change']]['Vehicle_ID'].unique()

    print(f"Found {len(lane_changing_vehicles):,} vehicles with lane changes")

    return lane_changing_vehicles

def extract_car_following_pairs(df, lane_changing_vehicles):
    """提取跟驰对"""
    print("Extracting car-following pairs...")

    # 筛选条件：
    # 1. 前车ID不为0 (有前车)
    # 2. 不是换道车辆
    # 3. 在同一车道 (通过Preceeding和当前车辆的Lane_ID匹配)

    # 首先，获取每条记录对应的前车车道
    # Preceeding字段存储的是前车ID

    # 创建前车信息映射
    vehicle_lanes = df[['Vehicle_ID', 'Lane_ID']].drop_duplicates()
    vehicle_lanes = vehicle_lanes.set_index('Vehicle_ID')['Lane_ID'].to_dict()

    # 检查前后车是否在同一车道
    df['preceding_lane'] = df['Preceeding'].map(vehicle_lanes)
    df['same_lane'] = df['Lane_ID'] == df['preceding_lane']

    # 筛选有效跟驰对
    # 条件1：前车ID > 0
    # 条件2：后车不在换道列表中
    # 条件3：同一车道
    valid_cf = (df['Preceeding'] > 0) & \
               (~df['Vehicle_ID'].isin(lane_changing_vehicles)) & \
               (df['same_lane'] == True)

    df_cf = df[valid_cf].copy()

    print(f"Found {len(df_cf):,} car-following records")

    return df_cf

def check_temporal_continuity(df_cf):
    """检查时空连续性，筛选共存时间超过阈值的配对"""
    print(f"Checking temporal continuity (min duration: {MIN_DURATION}s)...")

    # 为每个跟驰对（Vehicle_ID, Preceeding, Lane_ID）计算共存时间
    cf_pairs = df_cf.groupby(['Vehicle_ID', 'Preceeding', 'Lane_ID']).agg({
        'Global_Time': ['min', 'max', 'count']
    }).reset_index()

    cf_pairs.columns = ['Vehicle_ID', 'Preceeding', 'Lane_ID', 'time_min', 'time_max', 'frame_count']

    # 计算共存时间（毫秒转秒）
    cf_pairs['duration_s'] = (cf_pairs['time_max'] - cf_pairs['time_min']) / 1000.0

    # 筛选共存时间超过阈值的配对
    valid_pairs = cf_pairs[cf_pairs['duration_s'] >= MIN_DURATION]

    print(f"Valid pairs (>= {MIN_DURATION}s): {len(valid_pairs):,}")

    # 再次筛选数据
    valid_vehicle_ids = valid_pairs['Vehicle_ID'].values
    valid_preceeding_ids = valid_pairs['Preceeding'].values
    valid_lane_ids = valid_pairs['Lane_ID'].values

    # 使用复合条件筛选
    df_cf_valid = df_cf.copy()
    df_cf_valid['is_valid_pair'] = False

    for idx, row in df_cf_valid.iterrows():
        mask = (df_cf_valid.loc[idx, 'Vehicle_ID'] in valid_vehicle_ids) & \
               (df_cf_valid.loc[idx, 'Preceeding'] in valid_preceeding_ids) & \
               (df_cf_valid.loc[idx, 'Lane_ID'] in valid_lane_ids)

        # 这个方法太慢，让我用更高效的方式

    # 优化：直接合并
    df_cf_valid = df_cf.merge(
        valid_pairs[['Vehicle_ID', 'Preceeding', 'Lane_ID', 'duration_s']],
        on=['Vehicle_ID', 'Preceeding', 'Lane_ID'],
        how='inner'
    )

    print(f"After temporal filtering: {len(df_cf_valid):,} records")

    # 统计跟驰对数量
    n_pairs = df_cf_valid.groupby(['Vehicle_ID', 'Preceeding']).ngroups
    print(f"Total car-following pairs: {n_pairs:,}")

    return df_cf_valid, valid_pairs

def visualize_car_following(df_cf, valid_pairs):
    """可视化跟驰对分布"""
    print("Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 跟驰持续时间分布
    ax1 = axes[0, 0]
    durations = valid_pairs['duration_s']
    ax1.hist(durations, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=MIN_DURATION, color='red', linestyle='--', label=f'Min threshold ({MIN_DURATION}s)')
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Car-following Duration Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 每个车辆的跟驰次数
    ax2 = axes[0, 1]
    cf_counts = valid_pairs.groupby('Vehicle_ID').size()
    ax2.hist(cf_counts, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Number of CF pairs per vehicle')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of CF Pairs per Vehicle')
    ax2.grid(True, alpha=0.3)

    # 3. 车道分布
    ax3 = axes[1, 0]
    lane_counts = df_cf['Lane_ID'].value_counts().sort_index()
    ax3.bar(lane_counts.index, lane_counts.values, alpha=0.7, color='orange')
    ax3.set_xlabel('Lane ID')
    ax3.set_ylabel('Count')
    ax3.set_title('Car-following Distribution by Lane')
    ax3.grid(True, alpha=0.3)

    # 4. 跟驰时间分布（每对的帧数）
    ax4 = axes[1, 1]
    sample_pairs = valid_pairs.sample(min(5000, len(valid_pairs)), random_state=42)
    ax4.scatter(sample_pairs['duration_s'], sample_pairs['frame_count'],
               alpha=0.3, s=10)
    ax4.set_xlabel('Duration (s)')
    ax4.set_ylabel('Frame Count')
    ax4.set_title('Duration vs Frame Count')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'car_following_pairs.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {PIC_DIR}/car_following_pairs.png")

def generate_cf_report(df_cf, valid_pairs, original_count):
    """生成Markdown格式分析报告"""

    total_cf_records = len(df_cf)
    total_cf_pairs = valid_pairs.groupby(['Vehicle_ID', 'Preceeding']).ngroups
    avg_duration = valid_pairs['duration_s'].mean()
    max_duration = valid_pairs['duration_s'].max()

    report = f"""# 跟驰对提取分析报告

## 1. 概述

本报告描述了NGSIM-US-101数据集中跟驰对(Car-following Pairs)的提取过程和方法。

## 2. 提取方法

### 2.1 匹配规则

1. **唯一ID匹配**：根据 `Vehicle_ID` 和 `Preceeding` 提取前后车配对关系
2. **时空连续性检查**：
   - 前后车必须位于同一车道
   - 共存时间需超过 {MIN_DURATION} 秒
3. **异常剔除**：
   - 过滤掉存在换道行为的车辆

### 2.2 参数设置

| 参数 | 值 |
|------|------|
| 最小共存时间 | {MIN_DURATION} 秒 |
| 同车道要求 | {SAME_LANE} |

## 3. 数据统计

### 3.1 提取结果

| 指标 | 数值 |
|------|------|
| 原始记录数 | {original_count:,} |
| 跟驰记录数 | {total_cf_records:,} |
| 跟驰对数量 | {total_cf_pairs:,} |
| 提取比例 | {total_cf_records/original_count*100:.2f}% |

### 3.2 持续时间统计

| 指标 | 数值 |
|------|------|
| 平均持续时间 | {avg_duration:.2f} 秒 |
| 最大持续时间 | {max_duration:.2f} 秒 |
| 最小持续时间 | {valid_pairs['duration_s'].min():.2f} 秒 |

### 3.3 车道分布

| 车道 | 记录数 |
|------|--------|
"""

    # 添加车道分布
    lane_counts = df_cf['Lane_ID'].value_counts().sort_index()
    for lane_id, count in lane_counts.items():
        report += f"| {lane_id} | {count:,} |\n"

    report += f"""
## 4. 可视化结果

![跟驰对分布](./pic/car_following_pairs.png)

## 5. 结论

1. 成功提取了 {total_cf_pairs:,} 对跟驰关系
2. 平均跟驰持续时间为 {avg_duration:.2f} 秒
3. 数据已准备好用于后续特征计算

## 6. 输出文件

- 跟驰对数据: `code/output/car_following_pairs.csv`
- 可视化图像: `doc/pic/car_following_pairs.png`
- 分析报告: `doc/car_following_report.md`
"""

    report_path = os.path.join(DOC_DIR, 'car_following_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    return report

def main():
    """主函数"""
    global INPUT_FILE, OUTPUT_DIR, DOC_DIR, PIC_DIR

    parser = argparse.ArgumentParser(description='Car-following Pairs Extraction')
    parser.add_argument('--period', type=str, default=None,
                       help='Time period to process (e.g., "0750am-0805am")')
    parser.add_argument('--output-dir', type=str, default='code/output',
                       help='Output directory for data')
    parser.add_argument('--doc-dir', type=str, default='doc',
                       help='Output directory for docs')
    args = parser.parse_args()

    period = args.period
    OUTPUT_DIR = args.output_dir
    DOC_DIR = args.doc_dir
    PIC_DIR = os.path.join(DOC_DIR, 'pic')

    # 设置输入文件
    if period:
        INPUT_FILE = os.path.join('code/output', period, f'trajectories_cleaned_{period}.csv')
    else:
        INPUT_FILE = 'code/output/trajectories_cleaned.csv'

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PIC_DIR, exist_ok=True)

    title = "Step 3: Car-following Pairs Extraction"
    if period:
        title += f" - {period}"
    print("=" * 60)
    print(title)
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] Loading cleaned data...")
    original_count = len(pd.read_csv(INPUT_FILE, usecols=['Vehicle_ID']))
    df = load_cleaned_data()

    # 2. 检测换道行为
    print("\n[2/5] Detecting lane changes...")
    lane_changing_vehicles = detect_lane_changes(df)

    # 3. 提取跟驰对
    print("\n[3/5] Extracting car-following pairs...")
    df_cf = extract_car_following_pairs(df, lane_changing_vehicles)

    # 4. 时空连续性检查
    print("\n[4/5] Checking temporal continuity...")
    df_cf_valid, valid_pairs = check_temporal_continuity(df_cf)

    # 5. 可视化
    print("\n[5/5] Generating visualizations...")
    visualize_car_following(df_cf_valid, valid_pairs)

    # 6. 生成报告
    print("\nGenerating report...")
    generate_cf_report(df_cf_valid, valid_pairs, original_count)

    # 7. 保存数据
    if period:
        output_filename = f'car_following_pairs_{period}.csv'
    else:
        output_filename = 'car_following_pairs.csv'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df_cf_valid.to_csv(output_path, index=False)
    print(f"\nCar-following pairs saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Step 3 Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()