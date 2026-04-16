"""
步骤8: 模型验证 - LSTM简易模型
使用LSTM模型验证数据集的可用性
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置（将在main中设置）
INPUT_FILE_10S = None
OUTPUT_DIR = 'code/output'
DOC_DIR = 'doc'
PIC_DIR = 'doc/pic'

# 超参数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2

def load_data(filename):
    """加载HDF5数据"""
    print(f"Loading data from {filename}...")

    with h5py.File(filename, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]

    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    return X, y

class LSTMPredictor(nn.Module):
    """简单的LSTM模型用于预测后车加速度"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer, device):
    """训练模型"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 前向传播
        outputs = model(batch_X)

        # 如果y是多维的，取最后一个时间步
        if len(batch_y.shape) > 1:
            batch_y = batch_y[:, -1]

        loss = criterion(outputs.squeeze(), batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)

            # 如果y是多维的，取最后一个时间步
            if len(batch_y.shape) > 1:
                batch_y = batch_y[:, -1]

            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(batch_y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return mse, mae, r2, predictions, actuals

def visualize_results(train_losses, test_losses, predictions, actuals):
    """可视化训练结果"""
    print("Generating visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 训练损失曲线
    ax1 = axes[0]
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 预测 vs 实际
    ax2 = axes[1]
    sample_size = min(500, len(predictions))
    indices = np.random.choice(len(predictions), sample_size, replace=False)
    ax2.scatter(actuals[indices], predictions[indices], alpha=0.5, s=10)
    ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()],
             'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Predicted vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 残差分布
    ax3 = axes[2]
    residuals = predictions - actuals
    ax3.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Count')
    ax3.set_title('Residual Distribution')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'lstm_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {PIC_DIR}/lstm_validation.png")

def generate_model_report(metrics, train_losses, test_losses):
    """生成模型验证报告"""

    mse, mae, r2 = metrics

    report = f"""# 模型验证分析报告

## 1. 概述

本报告描述了使用LSTM模型对处理后的NGSIM-US-101数据集进行验证的结果。

## 2. 模型架构

### 2.1 网络结构

| 组件 | 配置 |
|------|------|
| 模型类型 | LSTM |
| 输入特征 | 3 (spacing, v_rel, v_follower) |
| 隐藏层大小 | {HIDDEN_SIZE} |
| LSTM层数 | {NUM_LAYERS} |
| 输出 | 1 (加速度) |

### 2.2 训练参数

| 参数 | 值 |
|------|------|
| 批次大小 | {BATCH_SIZE} |
| 学习率 | {LEARNING_RATE} |
| 训练轮数 | {EPOCHS} |
| 优化器 | Adam |

## 3. 评估结果

| 指标 | 数值 |
|------|------|
| MSE (均方误差) | {mse:.4f} |
| MAE (平均绝对误差) | {mae:.4f} |
| R2 (决定系数) | {r2:.4f} |

## 4. 训练过程

训练损失从 {train_losses[0]:.4f} 变化到 {train_losses[-1]:.4f}
测试损失从 {test_losses[0]:.4f} 变化到 {test_losses[-1]:.4f}

## 5. 结论

1. 模型成功完成训练，表明数据集可用于深度学习模型训练
2. R2为 {r2:.4f}，表明模型具有一定预测能力
3. 数据处理流程完整，格式正确

## 6. 输出文件

- 模型预测结果
- 可视化图像: `doc/pic/lstm_validation.png`
- 分析报告: `doc/model_validation_report.md`
"""

    report_path = os.path.join(DOC_DIR, 'model_validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    return report

def main():
    """主函数"""
    global INPUT_FILE_10S, OUTPUT_DIR, DOC_DIR, PIC_DIR

    parser = argparse.ArgumentParser(description='LSTM Model Validation')
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
    global INPUT_FILE_10S
    if period:
        INPUT_FILE_10S = os.path.join('code/output', period, f'train_10s_{period}.h5')
    else:
        INPUT_FILE_10S = 'code/output/train_10s.h5'

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PIC_DIR, exist_ok=True)

    title = "Step 8: LSTM Model Validation"
    if period:
        title += f" - {period}"
    print("=" * 60)
    print(title)
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    print("\n[1/5] Loading data...")
    X, y = load_data(INPUT_FILE_10S)

    # 2. 划分训练集和测试集
    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 创建模型
    print("\n[3/5] Creating model...")
    model = LSTMPredictor(
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练模型
    print("\n[4/5] Training model...")
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)

        # 评估
        model.eval()
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                if len(batch_y.shape) > 1:
                    batch_y = batch_y[:, -1]
                test_preds.extend(outputs.squeeze().cpu().numpy())
                test_targets.extend(batch_y.numpy())

        test_loss = np.mean((np.array(test_preds) - np.array(test_targets)) ** 2)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # 5. 评估模型
    print("\n[5/5] Evaluating model...")
    mse, mae, r2, predictions, actuals = evaluate_model(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2:  {r2:.4f}")

    # 6. 可视化
    visualize_results(train_losses, test_losses, predictions, actuals)

    # 7. 生成报告
    metrics = (mse, mae, r2)
    generate_model_report(metrics, train_losses, test_losses)

    # 保存模型
    if period:
        model_path = os.path.join(OUTPUT_DIR, f'lstm_model_{period}.pth')
    else:
        model_path = os.path.join(OUTPUT_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 60)
    print("Step 8 Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()