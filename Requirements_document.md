我们正在对NGSIM-US-101 数据集进行数据处理，下面是我们数据处理过程中的要求、流程以及预期效果。

注意：
1. 在处理数据的时候我们使用的语言统一为python语言
2. 使用conda中的py-310虚拟环境
3. 所有代码都放在 当前路径下的 ./code 中
4. 生成所有相关的Markdown文件都放在 ./doc 中
5. 生成所有相关的图片都放在 ./doc/pic 中，且生成报告的时候,记得引用 ./doc/pic 中的图片

数据集介绍:
1. 数据集存放的位置 './vehicle-trajectory-data'
2. 数据集的简单介绍:
> Vehicle_ID, Frame_ID, Total_Frames, Global_Time, Local_X, Local_Y, Global_X, Global_Y, v_Length, v_Width, v_Class, v_Vel, v_Acc, Lane_ID, O_Zone, D_Zone, Int_ID, Section_ID, Direction, Movement, Preceding, Following, Space_Headway, Time_Headway, Location

流程：
1. 轨迹平滑与去噪(Denoising)
(1) NGSIM 原始数据存在明显的定位抖动，直接计算速度和加速度会导致数据极其不平滑（甚至出现不符合物理逻辑的加速度）。
(2) 使用 Savitzky-Golay 滤波器。
(3) 在处理后, 检查加速度是否在合理范围内
(4) 要生成相关可视化的图像、生成相关Markdown报告 存放在 ./doc 中
(5) 将平滑去噪后的数据单独备份一下

2. 原始数据清洗与坐标转换
(1) 坐标对齐： 将 NGSIM 原始的 Local_Y(纵向位置)统一方向。
(2) 单位换算： 建议将原始的英尺(feet)和英里/小时(mph)统一转换为国际单位制(m, m/s, $m/s^2$)，便于后续模型计算。
(3) 要生成相关可视化的图像、生成相关Markdown报告 存放在 ./doc 中

3. 跟驰对(Car-following Pairs)提取
(1) 唯一 ID 匹配： 根据 Vehicle_ID 和 Preceding_ID 提取配对。
(2) 时空连续性检查： 确保两车在同一车道内且共存时间超过一定阈值（如 15s 以上）。
(3) 剔除异常： 过滤掉存在换道行为(Lane Change)的片段。
(4) 要生成相关可视化的图像、生成相关Markdown报告 存放在 ./doc 中

4. 特征变量计算
计算每一时刻的特征:
(1) 后车速度 ($v_{f}$): 直接提取平滑后的后车速度。
(2) 相对距离 ($s$): $s = y_{leader} - y_{follower} - L_{leader}$（注意必须减去前车车长 $L_{leader}$，得到净间距）。
(3) 相对速度 ($v_{rel}$): $v_{rel} = v_{leader} - v_{follower}$。
(4) 标签: 后车的加速度
(5) 要生成相关可视化的图像、生成相关Markdown报告 存放在 ./doc 中

5. 数据标准化与分段
(1) 滑动窗口切片： 将长轨迹切割成固定长度的时间序列片段 (10s 和 20s 一个 clip,这里要生成两个版本的)。
(2) 要生成相关可视化的图像、生成相关Markdown报告 存放在 ./doc 中

6. 将数据保存成 .h5 文件 (10s和20s两个版本的)

7. 对我们的提取的数据集进行可视化
(1) 对我们的数据进行高度的分析
(2) 生成丰富的可视化图像, 并生成相关的报告 存放在 ./doc 中

8. 创建建议的模型进行验证
(1) nn, LSTM简易模型进行验证
(2) 生成丰富的可视化图像, 并生成相关的报告 存放在 ./doc 中