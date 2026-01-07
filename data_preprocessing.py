"""
数据处理模块
负责加载、预处理UCI HAR数据集
"""

import numpy as np
import os


def load_data(data_path='UCI HAR Dataset'):
    """
    加载UCI HAR数据集
    
    参数:
        data_path: 数据集根目录路径
    
    返回:
        X_train, y_train, X_test, y_test
    """
    print("=" * 60)
    print("开始加载UCI HAR数据集...")
    print("=" * 60)
    
    # 训练集路径
    train_path = os.path.join(data_path, 'train')
    X_train_file = os.path.join(train_path, 'X_train.txt')
    y_train_file = os.path.join(train_path, 'y_train.txt')
    
    # 测试集路径
    test_path = os.path.join(data_path, 'test')
    X_test_file = os.path.join(test_path, 'X_test.txt')
    y_test_file = os.path.join(test_path, 'y_test.txt')
    
    # 检查文件是否存在
    required_files = [X_train_file, y_train_file, X_test_file, y_test_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}\n"
                                    f"请确保数据集已正确解压到 '{data_path}' 目录下")
    
    # 加载数据
    print(f"📂 从 {data_path} 加载数据...")
    X_train = np.loadtxt(X_train_file)
    y_train = np.loadtxt(y_train_file)
    X_test = np.loadtxt(X_test_file)
    y_test = np.loadtxt(y_test_file)
    
    print(f"✓ 训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"✓ 测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
    print(f"✓ 类别数量: {len(np.unique(y_train))} 类")
    
    return X_train, y_train, X_test, y_test


def get_activity_names():
    """
    获取活动类别名称
    
    返回:
        activity_names: 类别名称列表
    """
    return [
        'Walking',           # 1 - 走路
        'Walking Upstairs',  # 2 - 上楼
        'Walking Downstairs',# 3 - 下楼
        'Sitting',           # 4 - 坐
        'Standing',          # 5 - 站
        'Laying'             # 6 - 躺
    ]


def standardize_data(X_train, X_test):
    """
    Z-Score标准化
    关键步骤！必须进行标准化，否则梯度无法下降
    
    公式: x' = (x - μ) / σ
    
    参数:
        X_train: 训练集特征
        X_test: 测试集特征
    
    返回:
        X_train_std, X_test_std: 标准化后的数据
    """
    print("\n" + "=" * 60)
    print("进行数据标准化 (Z-Score Normalization)...")
    print("=" * 60)
    
    # 计算训练集的均值和标准差
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # 防止除以0
    std[std == 0] = 1.0
    
    # 标准化
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    
    print(f"✓ 训练集标准化完成")
    print(f"  - 均值范围: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  - 标准差范围: [{std.min():.4f}, {std.max():.4f}]")
    print(f"✓ 测试集标准化完成 (使用训练集统计量)")
    
    return X_train_std, X_test_std


def one_hot_encode(y, num_classes=6):
    """
    One-Hot编码
    将标签转换为one-hot向量
    
    例如: 1 -> [1, 0, 0, 0, 0, 0]
          3 -> [0, 0, 1, 0, 0, 0]
    
    参数:
        y: 标签数组 (值从1到6)
        num_classes: 类别总数
    
    返回:
        y_onehot: one-hot编码后的标签
    """
    y = y.astype(int) - 1  # 将标签从1-6转换为0-5
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


def split_validation(X_train, y_train, validation_split=0.15):
    """
    从训练集中分离出验证集
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        validation_split: 验证集比例
    
    返回:
        X_train_new, y_train_new, X_val, y_val
    """
    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    val_size = int(num_samples * validation_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train_new = X_train[train_indices]
    y_train_new = y_train[train_indices]
    
    return X_train_new, y_train_new, X_val, y_val


def analyze_data_distribution(y_train, y_test):
    """
    分析数据集类别分布
    
    参数:
        y_train: 训练集标签
        y_test: 测试集标签
    """
    print("\n" + "=" * 60)
    print("数据集类别分布分析")
    print("=" * 60)
    
    activity_names = get_activity_names()
    
    print("\n训练集分布:")
    for i in range(1, 7):
        count = np.sum(y_train == i)
        percentage = count / len(y_train) * 100
        print(f"  类别 {i} ({activity_names[i-1]:20s}): {count:4d} 样本 ({percentage:.2f}%)")
    
    print("\n测试集分布:")
    for i in range(1, 7):
        count = np.sum(y_test == i)
        percentage = count / len(y_test) * 100
        print(f"  类别 {i} ({activity_names[i-1]:20s}): {count:4d} 样本 ({percentage:.2f}%)")


def prepare_data(data_path='UCI HAR Dataset', validation_split=0.15):
    """
    完整的数据准备流程
    
    参数:
        data_path: 数据集路径
        validation_split: 验证集比例
    
    返回:
        X_train, y_train, X_val, y_val, X_test, y_test (均已标准化和one-hot编码)
    """
    # 1. 加载原始数据
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(data_path)
    
    # 2. 分析数据分布
    analyze_data_distribution(y_train_raw, y_test_raw)
    
    # 3. 标准化
    X_train_std, X_test_std = standardize_data(X_train_raw, X_test_raw)
    
    # 4. 分离验证集
    X_train, y_train, X_val, y_val = split_validation(
        X_train_std, y_train_raw, validation_split
    )
    
    print(f"\n✓ 训练/验证集分离完成:")
    print(f"  - 训练集: {X_train.shape[0]} 样本")
    print(f"  - 验证集: {X_val.shape[0]} 样本")
    print(f"  - 测试集: {X_test_std.shape[0]} 样本")
    
    # 5. One-Hot编码
    print("\n" + "=" * 60)
    print("进行One-Hot编码...")
    print("=" * 60)
    y_train_onehot = one_hot_encode(y_train)
    y_val_onehot = one_hot_encode(y_val)
    y_test_onehot = one_hot_encode(y_test_raw)
    
    print(f"✓ 标签编码完成")
    print(f"  - 原始标签范围: [1, 6]")
    print(f"  - One-Hot维度: {y_train_onehot.shape[1]}")
    print(f"  - 示例: 标签 1 -> {y_train_onehot[0]}")
    
    return X_train, y_train_onehot, X_val, y_val_onehot, X_test_std, y_test_onehot


if __name__ == "__main__":
    # 测试数据加载
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    print("\n✓ 数据准备完成！")
    print(f"  训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证集形状: X={X_val.shape}, y={y_val.shape}")
    print(f"  测试集形状: X={X_test.shape}, y={y_test.shape}")
