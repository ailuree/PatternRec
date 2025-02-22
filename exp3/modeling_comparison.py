import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # 数据建模方法
from sklearn.ensemble import RandomForestClassifier  # 算法建模方法
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 生成非线性数据
np.random.seed(42)
n_samples = 1000

# 创建螺旋形数据
def make_spiral_data(n_samples):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n_samples, 2)

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n_samples, 2)

    return np.vstack((x_a, x_b)), np.hstack((np.zeros(n_samples), np.ones(n_samples)))

# 可视化结果
def plot_results(X, y, X_test, y_test, data_model, algo_model, data_accuracy, algo_accuracy):
    plt.figure(figsize=(15, 5))
    
    # 1. 原始数据
    plt.subplot(131)
    plt.scatter(X[:n_samples, 0], X[:n_samples, 1], c='blue', label='类别 0')
    plt.scatter(X[n_samples:, 0], X[n_samples:, 1], c='red', label='类别 1')
    plt.title('原始数据分布')
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
    plt.title(f'数据建模(逻辑回归)\n准确率: {data_accuracy:.2f}')
    
    # 3. 算法建模预测结果
    plt.subplot(133)
    Z = algo_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8)
    plt.title(f'算法建模(随机森林)\n准确率: {algo_accuracy:.2f}')
    
    plt.tight_layout()
    plt.savefig('modeling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(y_test, data_pred, algo_pred):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    cm_data = confusion_matrix(y_test, data_pred)
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues')
    plt.title('数据建模（逻辑回归）混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    plt.subplot(122)
    cm_algo = confusion_matrix(y_test, algo_pred)
    sns.heatmap(cm_algo, annot=True, fmt='d', cmap='Blues')
    plt.title('算法建模（随机森林）混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主程序部分
if __name__ == "__main__":
    # 生成数据
    X, y = make_spiral_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. 数据建模方法 - 逻辑回归
    data_model = LogisticRegression()
    data_model.fit(X_train, y_train)
    data_pred = data_model.predict(X_test)
    data_accuracy = accuracy_score(y_test, data_pred)
    
    # 2. 算法建模方法 - 随机森林
    algo_model = RandomForestClassifier(n_estimators=100, random_state=42)
    algo_model.fit(X_train, y_train)
    algo_pred = algo_model.predict(X_test)
    algo_accuracy = accuracy_score(y_test, algo_pred)
    
    # 绘制结果
    plot_results(X, y, X_test, y_test, data_model, algo_model, data_accuracy, algo_accuracy)
    plot_confusion_matrices(y_test, data_pred, algo_pred)
    
    # 打印详细结果
    print("\n=== 模型性能对比 ===")
    print(f"数据建模（逻辑回归）准确率: {data_accuracy:.4f}")
    print(f"算法建模（随机森林）准确率: {algo_accuracy:.4f}")
