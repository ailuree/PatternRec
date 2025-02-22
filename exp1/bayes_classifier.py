import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据
data = np.array([
    [-5.01, -8.12, -3.68],
    [-5.43, -3.48, -3.54],
    [1.08, -5.52, 1.66],
    [0.86, -3.78, -4.11],
    [-2.67, 0.63, 7.39],
    [4.94, 3.29, 2.08],
    [-2.51, 2.09, -2.59],
    [-2.25, -2.13, -6.94],
    [5.56, 2.86, -2.26],
    [1.03, -3.33, 4.33],

    [-0.91, -0.18, -0.05],
    [1.30, -2.06, -3.53],
    [-7.75, -4.54, -0.95],
    [-5.47, 0.50, 3.92],
    [6.14, 5.72, -4.85],
    [3.60, 1.26, 4.36],
    [5.37, -4.63, -3.65],
    [7.18, 1.46, -6.66],
    [-7.39, 1.17, 6.30],
    [-7.50, -6.32, -0.31],

    [5.35, 2.26, 8.13],
    [5.12, 3.22, -2.66],
    [-1.34, -5.31, -9.87],
    [4.48, 3.42, 5.19],
    [7.11, 2.39, 9.21],
    [7.17, 4.33, -0.98],
    [5.75, 3.97, 6.65],
    [0.77, 0.27, 2.41],
    [0.90, -0.43, -8.71],
    [3.52, -0.36, 6.43]
])

class_1 = data[:10]    # ω₁
class_2 = data[10:20]  # ω₂

# 参数估计
def estimate_parameters(X):
    """
    参数估计 
    @param X: 样本数据
    @return: 均值向量和协方差矩阵
    """
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    return mean, cov

# 贝叶斯分类器
def bayes_classifier(x, mean1, cov1, mean2, cov2, p1, p2):
    """
    贝叶斯分类器
    @param x: 样本数据
    @param mean1: ω₁的均值向量
    @param cov1: ω₁的协方差矩阵
    @param mean2: ω₂的均值向量
    @param cov2: ω₂的协方差矩阵
    @param p1: ω₁的先验概率
    @param p2: ω₂的先验概率
    @return: 分类结果
    """
    prob1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1) * p1  # 计算样本x在ω₁类别下的概率  p(x|ω₁)P(ω₁) 即后验概率
    prob2 = multivariate_normal.pdf(x, mean=mean2, cov=cov2) * p2  # 计算样本x在ω₂类别下的概率  p(x|ω₂)P(ω₂) 
    return 1 if prob1 > prob2 else 2                               # 按照最大后验概率原则分类

# 分类样本封装函数
def classify_samples(X, mean1, cov1, mean2, cov2, p1, p2):
    """
    分类样本
    @param X: 样本数据
    @param mean1: ω₁的均值向量
    @param cov1: ω₁的协方差矩阵
    @param mean2: ω₂的均值向量
    @param cov2: ω₂的协方差矩阵
    @param p1: ω₁的先验概率
    @param p2: ω₂的先验概率
    @return: 分类结果
    """
    return [bayes_classifier(x, mean1, cov1, mean2, cov2, p1, p2) for x in X]

# 计算错误率
def calculate_error_rate(true_labels, predicted_labels):
    """
    计算错误率
    @param true_labels: 真实标签
    @param predicted_labels: 预测标签
    @return: 错误率  错误率就是分类错误的点的个数占总个数的百分比
    """
    return np.mean(np.array(true_labels) != np.array(predicted_labels))

# 可视化函数
def plot_decision_boundary(X, y, classifier, title, error_rate):
    """
    绘制二维决策边界
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = np.array([classifier(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.title(f"{title}\nError Rate: {error_rate:.2%}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Class')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

def plot_classification_results(X, y_true, y_pred, title, error_rate):
    """
    绘制二维分类结果
    """
    plt.figure(figsize=(12, 8))
    scatter_true = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap=plt.cm.RdYlBu, 
                               marker='o', s=100, edgecolor='black', label='True')
    scatter_pred = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.RdYlBu, 
                               marker='x', s=100, label='Predicted')
    plt.title(f"{title}\nError Rate: {error_rate:.2%}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=[scatter_true, scatter_pred], labels=['True', 'Predicted'])
    plt.colorbar(scatter_true, label='Class')
    
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            plt.annotate('', xy=(X[i, 0], X[i, 1]), xytext=(X[i, 0], X[i, 1]),
                         arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.show()

def plot_3d_classification(X, y_true, y_pred, title, error_rate):
    """
    绘制三维分类结果
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter_true = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap=plt.cm.RdYlBu, 
                              marker='o', s=100, label='True')
    scatter_pred = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap=plt.cm.RdYlBu, 
                              marker='x', s=100, label='Predicted')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(f"{title}\nError Rate: {error_rate:.2%}")
    
    plt.colorbar(scatter_true, label='Class')
    ax.legend()
    
    plt.show()

# 主要代码部分
if __name__ == "__main__":
    # 问题1: 两类问题
    # (1) 和 (2): 使用x1特征
    mean1, cov1 = estimate_parameters(class_1[:, 0].reshape(-1, 1))
    mean2, cov2 = estimate_parameters(class_2[:, 0].reshape(-1, 1))

    X = np.vstack((class_1[:, 0], class_2[:, 0])).reshape(-1, 1)
    true_labels = [1] * 10 + [2] * 10

    predicted_labels = classify_samples(X, mean1, cov1, mean2, cov2, 0.5, 0.5)
    error_rate = calculate_error_rate(true_labels, predicted_labels)

    print(f"使用x1特征的分类器错误率: {error_rate:.2%}")

    # (3): 使用x1和x2特征
    mean1_2d, cov1_2d = estimate_parameters(class_1[:, :2])
    mean2_2d, cov2_2d = estimate_parameters(class_2[:, :2])

    X_2d = np.vstack((class_1[:, :2], class_2[:, :2]))
    predicted_labels_2d = classify_samples(X_2d, mean1_2d, cov1_2d, mean2_2d, cov2_2d, 0.5, 0.5)
    error_rate_2d = calculate_error_rate(true_labels, predicted_labels_2d)

    print(f"使用x1和x2特征的分类器错误率: {error_rate_2d:.2%}")

    # 2特征的决策边界和分类结果
    plot_decision_boundary(X_2d, true_labels, 
                           lambda x: bayes_classifier(x, mean1_2d, cov1_2d, mean2_2d, cov2_2d, 0.5, 0.5),
                           'Decision Boundary (2 Features)', error_rate_2d)
    plot_classification_results(X_2d, true_labels, predicted_labels_2d, 
                                'Classification Results (2 Features)', error_rate_2d)

    # (4): 使用所有三个特征
    mean1_3d, cov1_3d = estimate_parameters(class_1)
    mean2_3d, cov2_3d = estimate_parameters(class_2)

    X_3d = np.vstack((class_1, class_2))
    predicted_labels_3d = classify_samples(X_3d, mean1_3d, cov1_3d, mean2_3d, cov2_3d, 0.5, 0.5)
    error_rate_3d = calculate_error_rate(true_labels, predicted_labels_3d)

    print(f"使用所有三个特征的分类器错误率: {error_rate_3d:.2%}")

    # 3特征的分类结果
    plot_3d_classification(X_3d, true_labels, predicted_labels_3d, 
                           'Classification Results (3 Features)', error_rate_3d)

    # 问题2: 三类问题
    class_3 = data[20:]
    
    # 三类问题的贝叶斯分类器
    def bayes_classifier_3class(x, means, covs, priors):
        probs = [multivariate_normal.pdf(x, mean=mean, cov=cov) * prior 
                 for mean, cov, prior in zip(means, covs, priors)]
        return np.argmax(probs) + 1

    def classify_samples_3class(X, means, covs, priors):
        return [bayes_classifier_3class(x, means, covs, priors) for x in X]

    # (1): 使用x1特征
    means_1d = [np.mean(class_i[:, 0]) for class_i in [class_1, class_2, class_3]]
    covs_1d = [np.cov(class_i[:, 0], rowvar=False) for class_i in [class_1, class_2, class_3]]
    X_1d = np.vstack((class_1[:, 0], class_2[:, 0], class_3[:, 0])).reshape(-1, 1)
    true_labels_3class = [1] * 10 + [2] * 10 + [3] * 10

    predicted_labels_1d = classify_samples_3class(X_1d, means_1d, covs_1d, [1/3, 1/3, 1/3])
    error_rate_1d = calculate_error_rate(true_labels_3class, predicted_labels_1d)

    print(f"三类问题使用x1特征的分类器错误率: {error_rate_1d:.2%}")

    # (2): 使用x1和x2特征
    means_2d = [np.mean(class_i[:, :2], axis=0) for class_i in [class_1, class_2, class_3]]
    covs_2d = [np.cov(class_i[:, :2], rowvar=False) for class_i in [class_1, class_2, class_3]]
    X_2d = np.vstack((class_1[:, :2], class_2[:, :2], class_3[:, :2]))

    predicted_labels_2d = classify_samples_3class(X_2d, means_2d, covs_2d, [1/3, 1/3, 1/3])
    error_rate_2d = calculate_error_rate(true_labels_3class, predicted_labels_2d)

    print(f"三类问题使用x1和x2特征的分类器错误率: {error_rate_2d:.2%}")

    # 三类问题分类使用2特征的分类结果
    # plot_classification_results(X_2d, true_labels_3class, predicted_labels_2d, 
    #                             'Classification Results (3 Classes, 2 Features)', error_rate_2d)

    # (3): 使用所有三个特征
    means_3d = [np.mean(class_i, axis=0) for class_i in [class_1, class_2, class_3]]
    covs_3d = [np.cov(class_i, rowvar=False) for class_i in [class_1, class_2, class_3]]
    X_3d = np.vstack((class_1, class_2, class_3))

    predicted_labels_3d = classify_samples_3class(X_3d, means_3d, covs_3d, [1/3, 1/3, 1/3])
    error_rate_3d = calculate_error_rate(true_labels_3class, predicted_labels_3d)

    print(f"三类问题使用所有三个特征的分类器错误率: {error_rate_3d:.2%}")

    # # 三类问题分类使用3特征的分类结果
    # plot_3d_classification(X_3d, true_labels_3class, predicted_labels_3d, 
    #                        'Classification Results (3 Classes, 3 Features)', error_rate_3d)