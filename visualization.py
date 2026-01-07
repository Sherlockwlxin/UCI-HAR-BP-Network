"""
可视化模块
用于绘制训练过程和结果分析图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_training_history(model_standard, model_momentum, save_path='training_comparison.png'):
    """
    对比绘制标准BP和Momentum BP的训练历史
    
    参数:
        model_standard: 标准BP模型
        model_momentum: Momentum BP模型
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_std = range(1, len(model_standard.train_loss_history) + 1)
    epochs_mom = range(1, len(model_momentum.train_loss_history) + 1)
    
    # 1. 训练集损失对比
    axes[0, 0].plot(epochs_std, model_standard.train_loss_history, 
                    label='Standard BP', color='#E74C3C', linewidth=2, alpha=0.8)
    axes[0, 0].plot(epochs_mom, model_momentum.train_loss_history, 
                    label='Momentum BP (α=0.9)', color='#3498DB', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加加速效果标注
    if len(model_momentum.train_loss_history) > 20:
        mid_point = len(model_momentum.train_loss_history) // 2
        axes[0, 0].annotate('Momentum加速效果', 
                           xy=(mid_point, model_momentum.train_loss_history[mid_point]),
                           xytext=(mid_point + 15, model_momentum.train_loss_history[mid_point] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2),
                           fontsize=10, color='green', fontweight='bold')
    
    # 2. 验证集损失对比
    axes[0, 1].plot(epochs_std, model_standard.val_loss_history, 
                    label='Standard BP', color='#E74C3C', linewidth=2, alpha=0.8)
    axes[0, 1].plot(epochs_mom, model_momentum.val_loss_history, 
                    label='Momentum BP (α=0.9)', color='#3498DB', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 训练集准确率对比
    axes[1, 0].plot(epochs_std, model_standard.train_acc_history, 
                    label='Standard BP', color='#E74C3C', linewidth=2, alpha=0.8)
    axes[1, 0].plot(epochs_mom, model_momentum.train_acc_history, 
                    label='Momentum BP (α=0.9)', color='#3498DB', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Training Accuracy', fontsize=12)
    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # 4. 验证集准确率对比
    axes[1, 1].plot(epochs_std, model_standard.val_acc_history, 
                    label='Standard BP', color='#E74C3C', linewidth=2, alpha=0.8)
    axes[1, 1].plot(epochs_mom, model_momentum.val_acc_history, 
                    label='Momentum BP (α=0.9)', color='#3498DB', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练历史对比图已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, activity_names, title='Confusion Matrix', 
                         save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签 (one-hot或整数)
        y_pred: 预测标签 (整数)
        activity_names: 活动名称列表
        title: 图表标题
        save_path: 保存路径
    """
    # 如果是one-hot编码，转换为整数标签
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 绘制
    plt.figure(figsize=(12, 10))
    
    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, yticklabels=activity_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()
    
    # 打印详细分析
    print("\n" + "=" * 60)
    print("混淆矩阵详细分析")
    print("=" * 60)
    for i, name in enumerate(activity_names):
        accuracy = cm_percent[i, i]
        print(f"\n{name}:")
        print(f"  - 类别准确率: {accuracy:.2f}%")
        
        # 找出最容易混淆的类别
        confused_with = []
        for j in range(len(activity_names)):
            if i != j and cm_percent[i, j] > 5:  # 混淆率超过5%
                confused_with.append(f"{activity_names[j]} ({cm_percent[i, j]:.2f}%)")
        
        if confused_with:
            print(f"  - 主要混淆: {', '.join(confused_with)}")
        else:
            print(f"  - 无明显混淆")


def plot_loss_curve_single(model, save_path='loss_curve.png'):
    """
    绘制单个模型的损失曲线
    
    参数:
        model: BP神经网络模型
        save_path: 保存路径
    """
    epochs = range(1, len(model.train_loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, model.train_loss_history, label='Training Loss', 
             color='#3498DB', linewidth=2, alpha=0.8)
    plt.plot(epochs, model.val_loss_history, label='Validation Loss', 
             color='#E74C3C', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 损失曲线已保存: {save_path}")
    plt.close()


def plot_accuracy_curve_single(model, save_path='accuracy_curve.png'):
    """
    绘制单个模型的准确率曲线
    
    参数:
        model: BP神经网络模型
        save_path: 保存路径
    """
    epochs = range(1, len(model.train_acc_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, model.train_acc_history, label='Training Accuracy', 
             color='#3498DB', linewidth=2, alpha=0.8)
    plt.plot(epochs, model.val_acc_history, label='Validation Accuracy', 
             color='#E74C3C', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 准确率曲线已保存: {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, activity_names):
    """
    打印分类报告
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        activity_names: 活动名称列表
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    print("\n" + "=" * 60)
    print("分类性能报告")
    print("=" * 60)
    
    for i, name in enumerate(activity_names):
        # 计算该类别的精确率、召回率、F1分数
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # 总体准确率
    accuracy = np.mean(y_true == y_pred)
    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")


def analyze_confusion_pairs(y_true, y_pred, activity_names):
    """
    分析最容易混淆的类别对
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        activity_names: 活动名称列表
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("最容易混淆的类别对")
    print("=" * 60)
    
    confusion_pairs = []
    for i in range(len(activity_names)):
        for j in range(i + 1, len(activity_names)):
            # 计算双向混淆总数
            confusion_count = cm[i, j] + cm[j, i]
            if confusion_count > 0:
                confusion_pairs.append((i, j, confusion_count))
    
    # 按混淆次数排序
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 显示前5对
    for idx, (i, j, count) in enumerate(confusion_pairs[:5], 1):
        print(f"{idx}. {activity_names[i]} ↔ {activity_names[j]}: {count} 次混淆")
        print(f"   - {activity_names[i]} 误判为 {activity_names[j]}: {cm[i, j]} 次")
        print(f"   - {activity_names[j]} 误判为 {activity_names[i]}: {cm[j, i]} 次")


if __name__ == "__main__":
    print("可视化模块测试")
    # 此模块需要与主程序配合使用
