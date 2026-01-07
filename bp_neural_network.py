"""
BP神经网络实现 - 手工实现版本
支持标准BP和Momentum BP两种训练方式
"""

import numpy as np


class BPNeuralNetwork:
    """
    三层BP神经网络：输入层 -> 隐藏层 -> 输出层
    完全基于NumPy手动实现，不依赖深度学习框架
    """
    
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.1, momentum=0.0, activation='relu'):
        """
        初始化神经网络
        
        参数:
            input_size: 输入层神经元数量 (561 for UCI HAR)
            hidden_size: 隐藏层神经元数量 (推荐64)
            output_size: 输出层神经元数量 (6 for UCI HAR)
            learning_rate: 学习率η
            momentum: 动量系数α (0.0为标准BP, 0.9为Momentum BP)
            activation: 隐藏层激活函数 ('relu' 或 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation_type = activation
        
        # He初始化权重 - 适合ReLU激活函数
        # Xavier初始化 - 适合Sigmoid激活函数
        if activation == 'relu':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        
        # 偏置初始化为0
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        # 动量项初始化
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
        # 记录训练历史
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
    
    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU导数"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 数值稳定性处理
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Sigmoid导数"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        """
        Softmax函数 - 输出层
        数值稳定性：减去最大值防止指数溢出
        """
        # 减去每行的最大值，防止exp溢出
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据 shape (batch_size, input_size)
        
        返回:
            output: Softmax输出 shape (batch_size, output_size)
        """
        # 输入层 -> 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # 隐藏层激活
        if self.activation_type == 'relu':
            self.a1 = self.relu(self.z1)
        else:
            self.a1 = self.sigmoid(self.z1)
        
        # 隐藏层 -> 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # 输出层Softmax
        self.output = self.softmax(self.z2)
        
        return self.output
    
    def cross_entropy_loss(self, y_true, y_pred):
        """
        交叉熵损失函数
        
        参数:
            y_true: 真实标签 (one-hot编码) shape (batch_size, output_size)
            y_pred: 预测概率 shape (batch_size, output_size)
        
        返回:
            loss: 平均损失值
        """
        m = y_true.shape[0]
        # 添加小值防止log(0)
        y_pred = np.clip(y_pred, 1e-10, 1.0)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward(self, X, y_true):
        """
        反向传播 - 核心算法
        
        参数:
            X: 输入数据 shape (batch_size, input_size)
            y_true: 真实标签 (one-hot编码) shape (batch_size, output_size)
        """
        m = X.shape[0]
        
        # 输出层误差：δ_output = y_pred - y_true (交叉熵+Softmax的优美性质)
        delta_output = self.output - y_true
        
        # 输出层梯度
        dW2 = np.dot(self.a1.T, delta_output) / m
        db2 = np.sum(delta_output, axis=0, keepdims=True) / m
        
        # 隐藏层误差：反向传播误差
        delta_hidden = np.dot(delta_output, self.W2.T)
        
        # 隐藏层激活函数导数
        if self.activation_type == 'relu':
            delta_hidden *= self.relu_derivative(self.z1)
        else:
            delta_hidden *= self.sigmoid_derivative(self.z1)
        
        # 隐藏层梯度
        dW1 = np.dot(X.T, delta_hidden) / m
        db1 = np.sum(delta_hidden, axis=0, keepdims=True) / m
        
        # 梯度裁剪，防止梯度爆炸
        max_grad_norm = 5.0
        dW1 = np.clip(dW1, -max_grad_norm, max_grad_norm)
        dW2 = np.clip(dW2, -max_grad_norm, max_grad_norm)
        db1 = np.clip(db1, -max_grad_norm, max_grad_norm)
        db2 = np.clip(db2, -max_grad_norm, max_grad_norm)
        
        # 更新权重 - 使用动量法
        # v_t = α * v_{t-1} - η * ∇J(W)
        # W_{t+1} = W_t + v_t
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * db1
        
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        """
        训练一个epoch (使用mini-batch)
        
        参数:
            X_train: 训练数据
            y_train: 训练标签 (one-hot)
            batch_size: 批大小
        
        返回:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        total_loss = 0
        correct = 0
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # 前向传播
            output = self.forward(X_batch)
            
            # 计算损失
            loss = self.cross_entropy_loss(y_batch, output)
            total_loss += loss * len(batch_indices)
            
            # 计算准确率
            predictions = np.argmax(output, axis=1)
            targets = np.argmax(y_batch, axis=1)
            correct += np.sum(predictions == targets)
            
            # 反向传播
            self.backward(X_batch, y_batch)
        
        avg_loss = total_loss / num_samples
        accuracy = correct / num_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, X, y):
        """
        评估模型
        
        参数:
            X: 测试数据
            y: 测试标签 (one-hot)
        
        返回:
            loss: 损失值
            accuracy: 准确率
            predictions: 预测结果
        """
        output = self.forward(X)
        loss = self.cross_entropy_loss(y, output)
        
        predictions = np.argmax(output, axis=1)
        targets = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == targets)
        
        return loss, accuracy, predictions
    
    def fit(self, X_train, y_train, X_val, y_val, 
            epochs=100, batch_size=32, verbose=True):
        """
        训练模型
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印训练信息
        """
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            
            # 验证集评估
            val_loss, val_acc, _ = self.evaluate(X_val, y_val)
            
            # 记录历史
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            
            # 打印信息
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入数据
        
        返回:
            predictions: 预测类别
        """
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions
