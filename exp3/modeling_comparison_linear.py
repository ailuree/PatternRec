import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_linear_data(n_samples=1000, noise=0.1):
    """生成线性可分的数据"""
    np.random.seed(42)
    
    # 生成两个类别的数据点
    X1 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(n_samples//2, 2))
    X2 = np.random.normal(loc=[-2, -2], scale=[1, 1], size=(n_samples//2, 2))
    
    # 合并数据
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples//2), np.ones(n_samples//2)))
    
    # 添加噪声
    X += np.random.normal(0, noise, X.shape)
    
    return X, y

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """评估模型性能，包括训练时间和预测时间"""
    # 训练时间
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 预测时间
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, train_time, predict_time, y_pred

def plot_results(X, y, X_test, y_test, data_model, algo_model, 
                data_metrics, algo_metrics, title="模型性能对比"):
    """可视化结果"""
    plt.figure(figsize=(15, 5))
    
    # 1. 原始数据
    plt.subplot(131)
    plt.scatter(X[:500, 0], X[:500, 1], c='blue', label='类别 0')
    plt.scatter(X[500:, 0], X[500:, 1], c='red', label='类别 1')
    plt.title('线性可分数据分布')
    plt.legend()
    
    # 2. 数据建模预测结果
    plt.subplot(132)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = data_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8)
    plt.title(f'数据建模(逻辑回归)\n准确率: {data_metrics[0]:.2f}\n训练时间: {data_metrics[1]:.4f}秒\n预测时间: {data_metrics[2]:.4f}秒')
    
    # 3. 算法建模预测结果
    plt.subplot(133)
    Z = algo_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8)
    plt.title(f'算法建模(随机森林)\n准确率: {algo_metrics[0]:.2f}\n训练时间: {algo_metrics[1]:.4f}秒\n预测时间: {algo_metrics[2]:.4f}秒')
    
    plt.tight_layout()
    plt.savefig('modeling_comparison_linear.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(y_test, data_pred, algo_pred, data_time, algo_time):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    cm_data = confusion_matrix(y_test, data_pred)
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues')
    plt.title(f'数据建模（逻辑回归）混淆矩阵\n计算时间: {data_time:.4f}秒')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    plt.subplot(122)
    cm_algo = confusion_matrix(y_test, algo_pred)
    sns.heatmap(cm_algo, annot=True, fmt='d', cmap='Blues')
    plt.title(f'算法建模（随机森林）混淆矩阵\n计算时间: {algo_time:.4f}秒')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_linear.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_model_details(data_metrics, algo_metrics):
    """打印模型详细信息"""
    print("\n=== 模型性能对比 ===")
    print(f"数据建模（逻辑回归）:")
    print(f"  - 准确率: {data_metrics[0]:.4f}")
    print(f"  - 训练时间: {data_metrics[1]:.4f}秒")
    print(f"  - 预测时间: {data_metrics[2]:.4f}秒")
    print(f"\n算法建模（随机森林）:")
    print(f"  - 准确率: {algo_metrics[0]:.4f}")
    print(f"  - 训练时间: {algo_metrics[1]:.4f}秒")
    print(f"  - 预测时间: {algo_metrics[2]:.4f}秒")

if __name__ == "__main__":
    # 生成线性可分的数据
    X, y = generate_linear_data(n_samples=1000, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. 数据建模 - 逻辑回归
    data_model = LogisticRegression()
    data_metrics = evaluate_model(data_model, X_train, X_test, y_train, y_test)
    
    # 2. 算法建模 - 随机森林 (使用较多的树来增加计算量)
    algo_model = RandomForestClassifier(n_estimators=200, random_state=42)
    algo_metrics = evaluate_model(algo_model, X_train, X_test, y_train, y_test)
    
    # 可视化结果
    plot_results(X, y, X_test, y_test, data_model, algo_model, data_metrics, algo_metrics)
    plot_confusion_matrices(y_test, data_metrics[3], algo_metrics[3], 
                          data_metrics[1] + data_metrics[2], 
                          algo_metrics[1] + algo_metrics[2])
    
    # 打印详细结果
    print_model_details(data_metrics, algo_metrics)
