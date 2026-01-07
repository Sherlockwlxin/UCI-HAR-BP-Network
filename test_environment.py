"""
快速测试脚本
用于验证环境配置和数据集是否正确
"""

import sys
import os


def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查 Python 版本...")
    print("=" * 60)
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✓ Python版本符合要求 (>= 3.7)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.7")
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("检查依赖包...")
    print("=" * 60)
    
    required_packages = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:15s} - 已安装")
        except ImportError:
            print(f"❌ {name:15s} - 未安装")
            all_installed = False
    
    if not all_installed:
        print("\n请运行以下命令安装缺失的包:")
        print("pip install numpy matplotlib seaborn scikit-learn")
    
    return all_installed


def check_dataset():
    """检查数据集"""
    print("\n" + "=" * 60)
    print("检查数据集...")
    print("=" * 60)
    
    data_path = 'UCI HAR Dataset'
    
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据集目录: {data_path}")
        print("\n请按以下步骤准备数据:")
        print("1. 下载: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
        print("2. 解压 'UCI HAR Dataset.zip'")
        print("3. 确保目录与main.py在同一级")
        return False
    
    print(f"✓ 找到数据集目录: {data_path}")
    
    # 检查必需文件
    required_files = [
        'train/X_train.txt',
        'train/y_train.txt',
        'test/X_test.txt',
        'test/y_test.txt'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        full_path = os.path.join(data_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ 缺失文件: {file_path}")
            all_files_exist = False
    
    return all_files_exist


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试数据加载...")
    print("=" * 60)
    
    try:
        import numpy as np
        
        data_path = 'UCI HAR Dataset'
        X_train = np.loadtxt(os.path.join(data_path, 'train/X_train.txt'))
        y_train = np.loadtxt(os.path.join(data_path, 'train/y_train.txt'))
        X_test = np.loadtxt(os.path.join(data_path, 'test/X_test.txt'))
        y_test = np.loadtxt(os.path.join(data_path, 'test/y_test.txt'))
        
        print(f"✓ 训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        print(f"✓ 测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")
        print(f"✓ 类别范围: {int(y_train.min())} - {int(y_train.max())}")
        
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("UCI HAR 实验环境检查")
    print("=" * 60 + "\n")
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 检查依赖包
    if not check_dependencies():
        return
    
    # 检查数据集
    if not check_dataset():
        return
    
    # 测试数据加载
    if not test_data_loading():
        return
    
    # 全部通过
    print("\n" + "=" * 60)
    print("✅ 环境检查完成！所有检查项通过！")
    print("=" * 60)
    print("\n现在可以运行主程序:")
    print("python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
