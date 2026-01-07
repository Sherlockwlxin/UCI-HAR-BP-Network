"""
主程序 - UCI HAR数据集人体行为识别实验
基于手工实现的BP神经网络

作者: [你的姓名]
日期: 2026年1月
课程: 机器学习
"""

import numpy as np
import time
import os
from bp_neural_network import BPNeuralNetwork
from data_preprocessing import prepare_data, get_activity_names
from visualization import (plot_training_history, plot_confusion_matrix, 
                          plot_loss_curve_single, plot_accuracy_curve_single,
                          print_classification_report, analyze_confusion_pairs)


def print_header(text):
    """打印美化的标题"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    """主函数"""
    print_header("UCI HAR 人体行为识别实验")
    print("基于改进BP算法的多层感知机实现")
    print("本实验完全基于NumPy手工实现，不依赖深度学习框架\n")
    
    # ========== 1. 设置随机种子 ==========
    np.random.seed(42)
    print("✓ 随机种子已设置: 42 (确保实验可复现)")
    
    # ========== 2. 数据准备 ==========
    print_header("步骤 1: 数据加载与预处理")
    
    data_path = 'UCI HAR Dataset'
    
    # 检查数据集是否存在
    if not os.path.exists(data_path):
        print(f"\n❌ 错误: 找不到数据集目录 '{data_path}'")
        print("\n请按以下步骤准备数据:")
        print("1. 下载数据集: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
        print("2. 解压 'UCI HAR Dataset.zip'")
        print("3. 确保目录结构如下:")
        print("   Project_Folder/")
        print("   |-- main.py")
        print("   |-- UCI HAR Dataset/")
        print("       |-- train/")
        print("       |-- test/")
        return
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
            data_path=data_path,
            validation_split=0.15
        )
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        return
    
    activity_names = get_activity_names()
    
    # ========== 3. 模型配置 ==========
    print_header("步骤 2: 模型配置")
    
    input_size = X_train.shape[1]  # 561
    hidden_size = 64               # 隐藏层神经元数
    output_size = 6                # 6个类别
    learning_rate = 0.01           # 降低学习率，避免梯度爆炸
    epochs = 100
    batch_size = 32
    
    print(f"网络结构: {input_size} → {hidden_size} (ReLU) → {output_size} (Softmax)")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    print(f"批大小: {batch_size}")
    
    # ========== 4. 实验组1: 标准BP算法 ==========
    print_header("步骤 3: 训练标准BP神经网络 (无动量)")
    
    model_standard = BPNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        momentum=0.0,  # 标准BP
        activation='relu'
    )
    
    print("开始训练...")
    start_time = time.time()
    
    model_standard.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    train_time_standard = time.time() - start_time
    print(f"\n✓ 标准BP训练完成! 耗时: {train_time_standard:.2f} 秒")
    
    # 评估标准BP
    _, test_acc_standard, y_pred_standard = model_standard.evaluate(X_test, y_test)
    print(f"✓ 测试集准确率: {test_acc_standard:.4f} ({test_acc_standard*100:.2f}%)")
    
    # ========== 5. 实验组2: Momentum BP算法 ==========
    print_header("步骤 4: 训练Momentum BP神经网络 (α=0.9)")
    
    model_momentum = BPNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        momentum=0.9,  # 引入动量
        activation='relu'
    )
    
    print("开始训练...")
    start_time = time.time()
    
    model_momentum.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    train_time_momentum = time.time() - start_time
    print(f"\n✓ Momentum BP训练完成! 耗时: {train_time_momentum:.2f} 秒")
    
    # 评估Momentum BP
    _, test_acc_momentum, y_pred_momentum = model_momentum.evaluate(X_test, y_test)
    print(f"✓ 测试集准确率: {test_acc_momentum:.4f} ({test_acc_momentum*100:.2f}%)")
    
    # ========== 6. 性能对比 ==========
    print_header("步骤 5: 实验结果对比")
    
    print("\n📊 性能对比表:")
    print("-" * 70)
    print(f"{'指标':<25} {'标准BP':<20} {'Momentum BP':<20}")
    print("-" * 70)
    print(f"{'训练时间 (秒)':<25} {train_time_standard:<20.2f} {train_time_momentum:<20.2f}")
    print(f"{'最终训练损失':<25} {model_standard.train_loss_history[-1]:<20.4f} {model_momentum.train_loss_history[-1]:<20.4f}")
    print(f"{'最终验证损失':<25} {model_standard.val_loss_history[-1]:<20.4f} {model_momentum.val_loss_history[-1]:<20.4f}")
    print(f"{'最终训练准确率':<25} {model_standard.train_acc_history[-1]:<20.4f} {model_momentum.train_acc_history[-1]:<20.4f}")
    print(f"{'最终验证准确率':<25} {model_standard.val_acc_history[-1]:<20.4f} {model_momentum.val_acc_history[-1]:<20.4f}")
    print(f"{'测试集准确率':<25} {test_acc_standard:<20.4f} {test_acc_momentum:<20.4f}")
    print("-" * 70)
    
    # 收敛速度分析
    print("\n📈 收敛速度分析:")
    threshold = 0.5
    epochs_to_converge_std = next((i for i, loss in enumerate(model_standard.train_loss_history) 
                                   if loss < threshold), epochs)
    epochs_to_converge_mom = next((i for i, loss in enumerate(model_momentum.train_loss_history) 
                                   if loss < threshold), epochs)
    
    print(f"达到损失<{threshold}所需轮数:")
    print(f"  - 标准BP: {epochs_to_converge_std} epochs")
    print(f"  - Momentum BP: {epochs_to_converge_mom} epochs")
    if epochs_to_converge_std > 0 and epochs_to_converge_mom < epochs_to_converge_std:
        speedup = (epochs_to_converge_std - epochs_to_converge_mom) / epochs_to_converge_std * 100
        print(f"  - 动量法加速: {speedup:.1f}% 更快!")
    elif epochs_to_converge_mom < epochs_to_converge_std:
        print(f"  - 动量法更快收敛")
    else:
        print(f"  - 标准BP收敛更快或两者相近")
    
    # ========== 7. 可视化 ==========
    print_header("步骤 6: 生成可视化图表")
    
    # 创建输出目录
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 7.1 训练历史对比
    plot_training_history(
        model_standard, 
        model_momentum,
        save_path=os.path.join(output_dir, 'training_comparison.png')
    )
    
    # 7.2 混淆矩阵 - Momentum BP (性能更好的模型)
    plot_confusion_matrix(
        y_test, 
        y_pred_momentum,
        activity_names,
        title='Confusion Matrix - Momentum BP',
        save_path=os.path.join(output_dir, 'confusion_matrix_momentum.png')
    )
    
    # 7.3 混淆矩阵 - 标准BP
    plot_confusion_matrix(
        y_test, 
        y_pred_standard,
        activity_names,
        title='Confusion Matrix - Standard BP',
        save_path=os.path.join(output_dir, 'confusion_matrix_standard.png')
    )
    
    # 7.4 单独的损失和准确率曲线 (Momentum BP)
    plot_loss_curve_single(
        model_momentum,
        save_path=os.path.join(output_dir, 'loss_curve_momentum.png')
    )
    
    plot_accuracy_curve_single(
        model_momentum,
        save_path=os.path.join(output_dir, 'accuracy_curve_momentum.png')
    )
    
    # ========== 8. 详细分类分析 ==========
    print_header("步骤 7: Momentum BP模型详细分析")
    
    print_classification_report(y_test, y_pred_momentum, activity_names)
    
    analyze_confusion_pairs(y_test, y_pred_momentum, activity_names)
    
    # ========== 9. 重点分析: Sitting vs Standing ==========
    print("\n" + "=" * 70)
    print("🔍 重点分析: 'Sitting' 和 'Standing' 的混淆情况")
    print("=" * 70)
    
    y_true_labels = np.argmax(y_test, axis=1)
    
    # Sitting (类别3) 被误判为 Standing (类别4) 的次数
    sitting_to_standing = np.sum((y_true_labels == 3) & (y_pred_momentum == 4))
    standing_to_sitting = np.sum((y_true_labels == 4) & (y_pred_momentum == 3))
    
    total_sitting = np.sum(y_true_labels == 3)
    total_standing = np.sum(y_true_labels == 4)
    
    print(f"\nSitting → Standing 误判: {sitting_to_standing}/{total_sitting} "
          f"({sitting_to_standing/total_sitting*100:.2f}%)")
    print(f"Standing → Sitting 误判: {standing_to_sitting}/{total_standing} "
          f"({standing_to_sitting/total_standing*100:.2f}%)")
    
    print("\n分析:")
    print("Sitting和Standing在传感器数据上非常相似,因为:")
    print("1. 两种状态都是静止的,加速度变化小")
    print("2. 重力方向相似,陀螺仪读数接近")
    print("3. 只有身体姿态的细微差异")
    
    # ========== 10. 总结 ==========
    print_header("实验总结")
    
    print("\n✅ 实验完成! 主要发现:")
    print("\n1. 动量法改进效果:")
    if epochs_to_converge_std > 0:
        speedup_pct = ((epochs_to_converge_std - epochs_to_converge_mom) / epochs_to_converge_std * 100)
        print(f"   - 收敛速度提升约 {speedup_pct:.1f}%")
    else:
        print(f"   - 收敛速度对比: 标准BP {epochs_to_converge_std} epochs vs Momentum BP {epochs_to_converge_mom} epochs")
    print(f"   - 测试准确率: {test_acc_momentum*100:.2f}% (vs 标准BP {test_acc_standard*100:.2f}%)")
    
    print("\n2. 模型性能:")
    print(f"   - 最终测试准确率: {test_acc_momentum*100:.2f}%")
    print(f"   - 高维特征 (561维) 下的良好泛化能力")
    
    print("\n3. 分类难点:")
    print("   - Sitting和Standing容易混淆 (静态姿态相似)")
    print("   - 动态活动 (Walking系列) 识别准确率较高")
    
    print(f"\n📁 所有结果已保存到 '{output_dir}' 目录:")
    print(f"   - training_comparison.png: 训练历史对比图")
    print(f"   - confusion_matrix_momentum.png: Momentum BP混淆矩阵")
    print(f"   - confusion_matrix_standard.png: 标准BP混淆矩阵")
    print(f"   - loss_curve_momentum.png: 损失曲线")
    print(f"   - accuracy_curve_momentum.png: 准确率曲线")
    
    print("\n" + "=" * 70)
    print("实验报告可以基于以上结果撰写!")
    print("=" * 70)


if __name__ == "__main__":
    main()
