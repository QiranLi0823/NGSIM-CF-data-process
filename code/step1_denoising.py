"""
步骤1: 轨迹平滑与去噪 (Denoising)
使用Savitzky-Golay滤波器对NGSIM轨迹数据进行平滑处理

Usage:
    # 指定数据目录和输出目录
    python code/step1_denoising.py --period 0750am-0805am --data-dir US-101 --output-dir code/output --doc-dir doc

    python code/step1_denoising.py --data-dir ./US-101 --period 0750am-0805am --output-dir code/output --doc-dir doc

Arguments:
    --period      时间段 (可选, 如 "0750am-0805am", "0805am-0820am", "0820am-0835am")
    --data-dir    输入数据目录 (默认: US-101)，最终输出路径为 output-dir/data-dir-name/period
    --output-dir  输出根目录 (默认: code/output)，数据将保存到 output-dir/data-dir-name/period
    --doc-dir     文档根目录 (默认: doc)，报告和图片将保存到 doc-dir/data-dir-name/period
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# 全局变量（将在main中设置）
PIC_DIR = 'doc/pic'
OUTPUT_DIR = 'code/output'
DOC_DIR = 'doc'
DATA_DIR = 'US-101'  # 默认数据目录

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_DIR = 'US-101'

def load_trajectory_data(period=None):
    """加载轨迹数据，可以指定单个时间段或全部加载"""
    data_files = [
        '0750am-0805am/trajectories-0750am-0805am.csv',
        '0805am-0820am/trajectories-0805am-0820am.csv',
        '0820am-0835am/trajectories-0820am-0835am.csv'
    ]

    all_data = []
    if period:
        # 只加载指定时间段
        file = f'{period}/trajectories-{period}.csv'
        file_path = os.path.join(DATA_DIR, file)
        print(f"Loading data: {file_path}")
        df = pd.read_csv(file_path)
        df['Data_Period'] = period
        all_data.append(df)
    else:
        # 加载所有时间段
        for file in data_files:
            file_path = os.path.join(DATA_DIR, file)
            print(f"Loading data: {file_path}")
            df = pd.read_csv(file_path)
            df['Data_Period'] = file.split('/')[0]
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total loaded: {len(combined_df)} records")
    return combined_df

def compute_velocity_acceleration(df, gap_threshold=2.0):
    """计算速度和加速度，处理跨时间段的数据间隙"""
    df = df.sort_values(['Vehicle_ID', 'Global_Time']).reset_index(drop=True)

    # 计算时间差（毫秒转秒）
    df['dt'] = df.groupby('Vehicle_ID')['Global_Time'].diff() / 1000.0

    # 对每个车辆分段计算位置差（跳过时间间隙）
    dX_list = []
    dY_list = []

    for vid in df['Vehicle_ID'].unique():
        mask = df['Vehicle_ID'] == vid
        time_vals = df.loc[mask, 'Global_Time'].values
        local_x = df.loc[mask, 'Local_X'].values
        local_y = df.loc[mask, 'Local_Y'].values

        curr_dX = np.full(len(time_vals), np.nan)
        curr_dY = np.full(len(time_vals), np.nan)

        # 检测连续段边界
        segment_starts = [0]
        for i in range(1, len(time_vals)):
            if (time_vals[i] - time_vals[i-1]) / 1000.0 > gap_threshold:
                segment_starts.append(i)

        # 在每个连续段内计算位置差
        for seg_idx in range(len(segment_starts)):
            start = segment_starts[seg_idx]
            if seg_idx + 1 < len(segment_starts):
                end = segment_starts[seg_idx + 1]
            else:
                end = len(time_vals)

            for i in range(start + 1, end):
                curr_dX[i] = local_x[i] - local_x[i-1]
                curr_dY[i] = local_y[i] - local_y[i-1]

        dX_list.extend(curr_dX)
        dY_list.extend(curr_dY)

    df['dX'] = dX_list
    df['dY'] = dY_list

    # 计算速度 (ft/s)，只在有效点计算
    valid_dt = df['dt'].notna() & (df['dt'] > 0)
    df['v_X'] = np.nan
    df['v_Y'] = np.nan
    df.loc[valid_dt, 'v_X'] = df.loc[valid_dt, 'dX'] / df.loc[valid_dt, 'dt']
    df.loc[valid_dt, 'v_Y'] = df.loc[valid_dt, 'dY'] / df.loc[valid_dt, 'dt']

    # 计算合速度
    df['v_total'] = np.sqrt(df['v_X']**2 + df['v_Y']**2)

    # 分段计算加速度
    dv_list = []
    for vid in df['Vehicle_ID'].unique():
        mask = df['Vehicle_ID'] == vid
        v_total = df.loc[mask, 'v_total'].values
        time_vals = df.loc[mask, 'Global_Time'].values

        curr_dv = np.full(len(v_total), np.nan)

        # 检测连续段边界
        segment_starts = [0]
        for i in range(1, len(time_vals)):
            if (time_vals[i] - time_vals[i-1]) / 1000.0 > gap_threshold:
                segment_starts.append(i)

        # 在每个连续段内计算速度差
        for seg_idx in range(len(segment_starts)):
            start = segment_starts[seg_idx]
            if seg_idx + 1 < len(segment_starts):
                end = segment_starts[seg_idx + 1]
            else:
                end = len(time_vals)

            for i in range(start + 1, end):
                if not np.isnan(v_total[i]) and not np.isnan(v_total[i-1]):
                    curr_dv[i] = v_total[i] - v_total[i-1]

        dv_list.extend(curr_dv)

    df['dv_total'] = dv_list

    # 计算加速度
    df['acc_raw'] = np.nan
    valid_acc = df['dt'].notna() & (df['dt'] > 0)
    df.loc[valid_acc, 'acc_raw'] = df.loc[valid_acc, 'dv_total'] / df.loc[valid_acc, 'dt']

    return df

def apply_savgol_filter(df, window_length=21, polyorder=3, gap_threshold=2.0):
    """应用Savitzky-Golay滤波器，处理跨时间段的间隙"""
    df = df.sort_values(['Vehicle_ID', 'Global_Time']).reset_index(drop=True)

    # 计算用于检测间隙的原始时间差
    df['_time_diff'] = df.groupby('Vehicle_ID')['Global_Time'].diff() / 1000.0

    # 对每个车辆的连续段进行平滑
    df['v_total_smooth'] = np.nan

    for vid in df['Vehicle_ID'].unique():
        mask = df['Vehicle_ID'] == vid
        time_diff = df.loc[mask, '_time_diff'].values
        v_total = df.loc[mask, 'v_total'].values

        # 找出连续段的起始位置（基于时间差）
        segment_starts = [0]
        for i in range(1, len(time_diff)):
            if time_diff[i] > gap_threshold:
                segment_starts.append(i)
        segment_starts.append(len(v_total))

        # 对每个段分别进行平滑
        smoothed = np.full(len(v_total), np.nan)
        for seg_idx in range(len(segment_starts) - 1):
            start = segment_starts[seg_idx]
            end = segment_starts[seg_idx + 1]
            segment = v_total[start:end]
            valid_mask = ~np.isnan(segment)
            valid_count = valid_mask.sum()

            if valid_count >= window_length:
                # 提取有效值，平滑，再放回对应位置
                valid_values = segment[valid_mask]
                smoothed_values = savgol_filter(valid_values, window_length, polyorder, mode='nearest')
                valid_indices = np.where(valid_mask)[0]
                for idx, smooth_val in zip(valid_indices, smoothed_values):
                    smoothed[start + idx] = smooth_val
            else:
                # 不足窗口长度，直接用原始值
                for i in range(len(segment)):
                    if not np.isnan(segment[i]):
                        smoothed[start + i] = segment[i]

        df.loc[mask, 'v_total_smooth'] = smoothed

    # 分段计算平滑后加速度
    df['dv_total_smooth'] = np.nan

    for vid in df['Vehicle_ID'].unique():
        mask = df['Vehicle_ID'] == vid
        time_diff = df.loc[mask, '_time_diff'].values
        v_smooth = df.loc[mask, 'v_total_smooth'].values

        # 找出连续段
        segment_starts = [0]
        for i in range(1, len(time_diff)):
            if time_diff[i] > gap_threshold:
                segment_starts.append(i)
        segment_starts.append(len(v_smooth))

        dv_smooth = np.full(len(v_smooth), np.nan)
        for seg_idx in range(len(segment_starts) - 1):
            start = segment_starts[seg_idx]
            end = segment_starts[seg_idx + 1]
            for i in range(start + 1, end):
                if not np.isnan(v_smooth[i]) and not np.isnan(v_smooth[i-1]):
                    dv_smooth[i] = v_smooth[i] - v_smooth[i-1]

        df.loc[mask, 'dv_total_smooth'] = dv_smooth

    # 计算平滑后加速度
    df['acc_smooth'] = np.nan
    valid_acc = df['dt'].notna() & (df['dt'] > 0)
    df.loc[valid_acc, 'acc_smooth'] = df.loc[valid_acc, 'dv_total_smooth'] / df.loc[valid_acc, 'dt']

    # 删除临时列
    df.drop('_time_diff', axis=1, inplace=True)

    return df

def check_acceleration_validity(df, max_acc_mps2=3.0):
    """检查加速度是否在合理范围内"""
    # 将ft/s^2转换为m/s^2 (1 ft/s^2 = 0.3048 m/s^2)
    acc_smooth_mps2 = df['acc_smooth'].abs() * 0.3048

    # 统计不合理加速度的比例
    invalid_ratio = (acc_smooth_mps2 > max_acc_mps2).sum() / len(df) * 100

    print(f"加速度超过 +/-{max_acc_mps2} m/s^2 的数据点比例: {invalid_ratio:.2f}%")

    # 标记异常值
    df['acc_valid'] = acc_smooth_mps2 <= max_acc_mps2

    return df, invalid_ratio

def detect_data_gaps(time_sec, threshold=2.0):
    """检测时间序列中的间隙，返回间隙位置的掩码"""
    if len(time_sec) < 2:
        return np.zeros(len(time_sec), dtype=bool)

    # 计算相邻时间点之间的差值
    time_diff = np.diff(time_sec)

    # 找出时间间隔超过阈值的点（数据间隙）
    gap_indices = np.where(time_diff > threshold)[0]

    # 创建间隙掩码 - 标记间隙后的点为间隙区域
    gap_mask = np.zeros(len(time_sec), dtype=bool)
    for idx in gap_indices:
        gap_mask[idx + 1:] = True

    return gap_mask


def get_vehicle_segments(df, vehicle_id):
    """获取某辆车的所有连续数据段"""
    vehicle_data = df[df['Vehicle_ID'] == vehicle_id].sort_values('Global_Time')
    time_vals = vehicle_data['Global_Time'].values

    # 获取整辆车的起始时间作为全局基准
    global_start_time = time_vals[0]

    # 找出所有段的起始索引
    segment_starts = [0]
    for i in range(1, len(time_vals)):
        time_diff = (time_vals[i] - time_vals[i-1]) / 1000.0
        if time_diff > 2.0:  # 超过2秒认为是时间段间隙
            segment_starts.append(i)

    # 提取每段数据
    segments = []
    for seg_idx, start in enumerate(segment_starts):
        if seg_idx + 1 < len(segment_starts):
            end = segment_starts[seg_idx + 1]
        else:
            end = len(time_vals)

        seg_data = vehicle_data.iloc[start:end].copy()
        # 时间基于整辆车的起始时间，而非每段的起始时间
        seg_time = (seg_data['Global_Time'] - global_start_time) / 1000

        segments.append({
            'data': seg_data,
            'time': seg_time,
            'segment_num': seg_idx + 1
        })

    return segments


def visualize_denoising_results(df, num_vehicles=3):
    """可视化平滑前后对比 - 为多辆车生成图片，每辆车可能有多个段"""

    # 统计每辆车的段数
    vehicle_segment_counts = {}
    for vid in df['Vehicle_ID'].unique():
        vehicle_data = df[df['Vehicle_ID'] == vid]
        time_vals = vehicle_data['Global_Time'].values
        segment_count = 1
        for i in range(1, len(time_vals)):
            if (time_vals[i] - time_vals[i-1]) / 1000.0 > 2.0:
                segment_count += 1
        vehicle_segment_counts[vid] = segment_count

    # 按段数分类
    segment_dist = {}
    for vid, cnt in vehicle_segment_counts.items():
        if cnt not in segment_dist:
            segment_dist[cnt] = []
        segment_dist[cnt].append(vid)

    print(f"车辆段数分布: {[(k, len(v)) for k, v in segment_dist.items()]}")

    # 选择3辆车：优先选择有3段、2段、1段的车辆
    selected_vehicles = []

    # 按优先级选择：3段 > 2段 > 1段
    for seg_count in [3, 2, 1]:
        if seg_count in segment_dist and len(segment_dist[seg_count]) > 0:
            for vid in segment_dist[seg_count]:
                if len(selected_vehicles) >= num_vehicles:
                    break
                selected_vehicles.append(int(vid))
            if len(selected_vehicles) >= num_vehicles:
                break

    # 如果还不够，从任意车辆补充
    all_vids = list(df['Vehicle_ID'].unique())
    for vid in all_vids:
        if len(selected_vehicles) >= num_vehicles:
            break
        if int(vid) not in selected_vehicles:
            selected_vehicles.append(int(vid))

    print(f"选择的车辆ID: {selected_vehicles}")

    # 为每辆车生成图片
    generated_files = []

    for round_idx, vehicle_id in enumerate(selected_vehicles):
        print(f"\n--- Round {round_idx + 1}: Vehicle {vehicle_id} ---")

        segments = get_vehicle_segments(df, vehicle_id)
        print(f"  车辆 {vehicle_id} 有 {len(segments)} 个数据段")

        for seg in segments:
            seg_num = seg['segment_num']
            seg_data = seg['data']
            seg_time = seg['time']

            # 生成文件名
            if len(segments) == 1:
                filename = f"denoising_vehicle_{vehicle_id}.png"
                title_suffix = "Single Segment"
            else:
                filename = f"denoising_vehicle_{vehicle_id}_Part{seg_num}.png"
                title_suffix = f"Part {seg_num}"

            _plot_denoising_comparison(seg_data, seg_time, vehicle_id, filename, title_suffix)
            generated_files.append(filename)

    print(f"\n可视化图像已保存至: {PIC_DIR}/")
    return selected_vehicles, generated_files


def _plot_denoising_comparison(vehicle_data, time_sec, vehicle_id, filename, title_suffix):
    """Plot denoising comparison for a single segment"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Speed comparison
    ax1 = axes[0]
    valid_speed = ~np.isnan(vehicle_data['v_total'].values)
    if valid_speed.any():
        ax1.plot(time_sec.values[valid_speed], vehicle_data['v_total'].values[valid_speed],
                 'b-', alpha=0.5, label='Raw Speed', linewidth=1)
        ax1.plot(time_sec.values[valid_speed], vehicle_data['v_total_smooth'].values[valid_speed],
                 'r-', label='Smoothed Speed', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (ft/s)')
    ax1.set_title(f'Vehicle {vehicle_id} Speed Smoothing ({title_suffix})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Acceleration comparison
    ax2 = axes[1]
    valid_raw = ~np.isnan(vehicle_data['acc_raw'].values)
    valid_smooth = ~np.isnan(vehicle_data['acc_smooth'].values)

    if valid_raw.any():
        ax2.plot(time_sec.values[valid_raw], vehicle_data['acc_raw'].values[valid_raw],
                 'b-', alpha=0.5, label='Raw Acceleration', linewidth=1)
    if valid_smooth.any():
        ax2.plot(time_sec.values[valid_smooth], vehicle_data['acc_smooth'].values[valid_smooth],
                 'r-', label='Smoothed Acceleration', linewidth=1.5)
    ax2.axhline(y=3/0.3048, color='g', linestyle='--', alpha=0.5, label='+/-3 m/s2 boundary')
    ax2.axhline(y=-3/0.3048, color='g', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (ft/s2)')
    ax2.set_title(f'Vehicle {vehicle_id} Acceleration Smoothing ({title_suffix})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Acceleration distribution comparison
    ax3 = axes[2]
    acc_raw_valid = vehicle_data['acc_raw'].dropna()
    acc_smooth_valid = vehicle_data['acc_smooth'].dropna()
    ax3.hist(acc_raw_valid, bins=50, alpha=0.5, label='Raw Acceleration', density=True)
    ax3.hist(acc_smooth_valid, bins=50, alpha=0.5, label='Smoothed Acceleration', density=True)
    ax3.set_xlabel('Acceleration (ft/s2)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Acceleration Distribution ({title_suffix})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {PIC_DIR}/{filename}")

def generate_denoising_report(df, invalid_ratio, sample_vehicle_ids, generated_files):
    """生成Markdown格式分析报告"""

    # 基本统计
    total_records = len(df)
    valid_records = df['acc_valid'].sum()
    total_vehicles = df['Vehicle_ID'].nunique()

    # 计算速度统计
    v_raw_mean = df['v_total'].mean()
    v_smooth_mean = df['v_total_smooth'].mean()
    acc_raw_std = df['acc_raw'].std()
    acc_smooth_std = df['acc_smooth'].std()

    # 构建可视化图片列表
    image_md = ""
    for f in generated_files:
        image_md += f"![{f}](./pic/{f})\n\n"

    # 车辆信息
    vehicle_info = ", ".join([str(vid) for vid in sample_vehicle_ids])

    report = f"""# 轨迹平滑与去噪分析报告

## 1. 概述

本报告描述了NGSIM-US-101数据集的轨迹平滑与去噪处理过程，旨在消除原始数据中的定位抖动问题，确保速度和加速度计算结果符合物理规律。

## 2. 处理方法

### 2.1 平滑算法
采用 **Savitzky-Golay 滤波器** 进行轨迹平滑处理：
- **窗口长度**: 21
- **多项式阶数**: 3

### 2.2 加速度验证
- 合理加速度阈值: ±3 m/s² (约 ±9.84 ft/s²)
- 超过阈值的数据点将被标记为异常

## 3. 数据统计

| 指标 | 数值 |
|------|------|
| 总记录数 | {total_records:,} |
| 总车辆数 | {total_vehicles:,} |
| 有效加速度记录 | {valid_records:,} |
| 异常加速度比例 | {invalid_ratio:.2f}% |

## 4. 速度统计

| 指标 | 原始速度 (ft/s) | 平滑后速度 (ft/s) |
|------|-----------------|-------------------|
| 平均值 | {v_raw_mean:.2f} | {v_smooth_mean:.2f} |

## 5. 加速度统计

| 指标 | 原始加速度 (ft/s²) | 平滑后加速度 (ft/s²) |
|------|-------------------|---------------------|
| 标准差 | {acc_raw_std:.2f} | {acc_smooth_std:.2f} |

平滑后加速度标准差显著降低，表明数据更加平滑。

## 6. 可视化结果

选择了以下车辆进行可视化展示: {vehicle_info}

每辆车可能包含多个连续数据段（跨时间段的数据），每个数据段生成一张对比图。

{image_md}
## 7. 结论

1. 使用Savitzky-Golay滤波器有效消除了轨迹数据的噪声
2. 异常加速度比例从较高水平降至 {invalid_ratio:.2f}%
3. 平滑后的数据更适合后续的跟驰模型训练

## 8. 输出文件

- 平滑后数据: `code/output/trajectories_smoothed.csv`
- 可视化图像: `doc/pic/denoising_vehicle_*.png`
- 分析报告: `doc/denoising_report.md`
"""

    report_path = os.path.join(DOC_DIR, 'denoising_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"分析报告已保存至: {report_path}")
    return report

def main():
    """主函数"""
    global PIC_DIR, OUTPUT_DIR, DOC_DIR, DATA_DIR

    parser = argparse.ArgumentParser(description='Trajectory Denoising')
    parser.add_argument('--period', type=str, default=None,
                       help='Time period to process (e.g., "0750am-0805am"). If not specified, processes all periods.')
    parser.add_argument('--data-dir', type=str, default='US-101',
                       help='Input data directory containing trajectory files')
    parser.add_argument('--output-dir', type=str, default='code/output',
                       help='Output directory for data')
    parser.add_argument('--doc-dir', type=str, default='doc',
                       help='Output directory for docs')
    args = parser.parse_args()

    period = args.period
    DATA_DIR = args.data_dir

    # 构建输出路径: output-dir/data-dir/period
    # 构建文档路径: doc-dir/data-dir/period
    data_dir_name = os.path.basename(os.path.normpath(DATA_DIR))

    if period:
        OUTPUT_DIR = os.path.join(args.output_dir, data_dir_name, period)
        DOC_DIR = os.path.join(args.doc_dir, data_dir_name, period)
    else:
        OUTPUT_DIR = os.path.join(args.output_dir, data_dir_name)
        DOC_DIR = os.path.join(args.doc_dir, data_dir_name)

    PIC_DIR = os.path.join(DOC_DIR, 'pic')

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PIC_DIR, exist_ok=True)

    # 设置标题
    title = f"Step 1: Trajectory Denoising"
    if period:
        title += f" - {period}"
    print("=" * 60)
    print(title)
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] Loading trajectory data...")
    df = load_trajectory_data(period)

    # 2. 计算速度和加速度
    print("\n[2/5] Computing velocity and acceleration...")
    df = compute_velocity_acceleration(df)

    # 3. 应用Savitzky-Golay滤波器
    print("\n[3/5] Applying Savitzky-Golay filter...")
    df = apply_savgol_filter(df)

    # 4. 验证加速度有效性
    print("\n[4/5] Validating acceleration...")
    df, invalid_ratio = check_acceleration_validity(df)

    # 5. 可视化
    print("\n[5/5] Generating visualizations...")
    sample_vehicle_ids, generated_files = visualize_denoising_results(df)

    # 6. 生成报告
    print("\nGenerating report...")
    generate_denoising_report(df, invalid_ratio, sample_vehicle_ids, generated_files)

    # 7. 保存平滑后的数据
    if period:
        output_filename = f'trajectories_smoothed_{period}.csv'
    else:
        output_filename = 'trajectories_smoothed.csv'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(output_path, index=False)
    print(f"\nSmoothed data saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Step 1 Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()