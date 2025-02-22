import numpy as np
from scipy.stats import multivariate_normal     # 引入多元正态分布计算函数
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据加载   这里加载时，前10个样本是类别1，中间10个是类别2，最后是类别3，每个类别的样本点的三个特征存储在一个列表中
data = np.array([
    [-5.01, -8.12, -3.68], [-5.43, -3.48, -3.54], [1.08, -5.52, 1.66],
    [0.86, -3.78, -4.11], [-2.67, 0.63, 7.39], [4.94, 3.29, 2.08],
    [-2.51, 2.09, -2.59], [-2.25, -2.13, -6.94], [5.56, 2.86, -2.26],
    [1.03, -3.33, 4.33], [-0.91, -0.18, -0.05], [1.30, -2.06, -3.53],
    [-7.75, -4.54, -0.95], [-5.47, 0.50, 3.92], [6.14, 5.72, -4.85],
    [3.60, 1.26, 4.36], [5.37, -4.63, -3.65], [7.18, 1.46, -6.66],
    [-7.39, 1.17, 6.30], [-7.50, -6.32, -0.31], [5.35, 2.26, 8.13],
    [5.12, 3.22, -2.66], [-1.34, -5.31, -9.87], [4.48, 3.42, 5.19],
    [7.11, 2.39, 9.21], [7.17, 4.33, -0.98], [5.75, 3.97, 6.65],
    [0.77, 0.27, 2.41], [0.90, -0.43, -8.71], [3.52, -0.36, 6.43]
])

# 将数据分为三类
class_1, class_2, class_3 = data[:10], data[10:20], data[20:]    ##前10个样本为𝜔1，中间10个是𝜔2，最后是ω3

def estimate_parameters(X):
    """
    参数估计
    @param X: 样本数据
    @return: 均值和协方差矩阵
    """
    return np.mean(X, axis=0), np.cov(X, rowvar=False)       # 返回输入样本的均值和协方差矩阵 rowvar=False表示每一列代表一个特征  输入的X可以是一维、二维和三维数组

def calculate_error_rate(true_labels, predicted_labels):
    """
    计算分类错误率
    @param true_labels: 真实标签
    @param predicted_labels: 预测标签
    @return: 错误率  这个错误率就是分类错误的样本数除以总的样本数
    """
    return np.mean(np.array(true_labels) != np.array(predicted_labels))  # 计算错误率 np.array(true_labels) != np.array(predicted_labels)返回一个布尔数组，True表示分类错误，False表示分类正确  np.mean()计算True的比例

def bayes_classifier(x, means, covs, priors):
    """
    贝叶斯分类器  
    输入一个样本点特征，之后计算这个样本点属于每个类别的后验概率（条件概率×先验概率），返回概率最大的类别的标签
    @param x: 样本点，即特征向量，可以是一维、二维或三维
    @param means: 各类别的均值向量
    @param covs: 各类别的协方差矩阵
    @param priors: 各类别的先验概率
    @return: 预测的类别标签
    """
    probs = [multivariate_normal.pdf(x, mean=mean, cov=cov) * prior 
             for mean, cov, prior in zip(means, covs, priors)]       # 用多元正态分布的概率密度函数计算每个类别的概率  probs列表包含每个类别的后验概率
    return np.argmax(probs) + 1  # 返回概率最大的类别的标签  np.argmax(probs)返回的是 probs 列表中最大值的索引  +1后，类别标签被定为1、2或者1、2、3

def classify_samples(X, means, covs, priors):
    """
    对多个样本进行分类
    @param X: 总的样本数据
    @param means: 各类别的均值向量
    @param covs: 各类别的协方差矩阵
    @param priors: 各类别的先验概率
    @return: 预测的类别标签数组
    """
    return np.array([bayes_classifier(x, means, covs, priors) for x in X])  # 从X中取出每个样本，对每个样本进行分类   返回一个数组，数组中的每个元素是一个样本的类别标签

# 绘图
class Visualizer:
    """
    可视化类，用于绘制决策边界和错误率
    """
    @staticmethod
    def plot_decision_boundary(X, y, means, covs, priors, error_rate, dim):
        """
        绘制决策边界
        """
        plt.figure(figsize=(10, 6) if dim == 1 else (10, 8))  # 设置图像大小
        colors = ['r', 'b', 'g']  # 颜色
        labels = ['类别1', '类别2', '类别3']  # 标签
        
        y = np.array(y)
        
        if dim == 1:
            Visualizer._plot_1d(X, y, means, covs, priors, colors, labels)
        else:
            Visualizer._plot_2d(X, y, means, covs, priors, colors, labels)
        
        plt.title(f"{'二' if len(means) == 2 else '三'}类问题：{dim}维特征的决策边界 (错误率: {error_rate:.2%})")
        plt.xlabel("x1")
        plt.ylabel("x2" if dim > 1 else "")
        plt.legend()
        plt.show()

    @staticmethod
    def _plot_1d(X, y, means, covs, priors, colors, labels):
        """
        绘制一维决策边界
        """
        for i, (color, label) in enumerate(zip(colors, labels)):
            class_data = X[y == i+1]
            plt.scatter(class_data, np.zeros_like(class_data), c=color, label=label)
        
        x_range = np.linspace(X.min() - 1, X.max() + 1, 1000).reshape(-1, 1)
        y_pred = classify_samples(x_range, means, covs, priors)
        
        for i, color in enumerate(colors):
            plt.plot(x_range[y_pred == i+1], [(i-1)*0.1] * np.sum(y_pred == i+1), f'{color}.', alpha=0.1)
        
        plt.ylim(-0.5, 0.5)
        plt.yticks([])

    @staticmethod
    def _plot_2d(X, y, means, covs, priors, colors, labels):
        """
        绘制二维决策边界
        """
        for i, (color, label) in enumerate(zip(colors, labels)):
            class_data = X[y == i+1]
            plt.scatter(class_data[:, 0], class_data[:, 1], c=color, label=label)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = classify_samples(np.c_[xx.ravel(), yy.ravel()], means, covs, priors)
        Z = Z.reshape(xx.shape)
        
        cmap = plt.cm.RdYlBu if len(means) == 2 else plt.cm.viridis
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
        plt.contour(xx, yy, Z, colors='k', linewidths=0.5)

    @staticmethod
    def plot_decision_boundary_3d(X, y, means, covs, priors, error_rate, class_count):
        """
        绘制三维决策边界
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'b', 'g']
        labels = ['类别1', '类别2', '类别3'][:class_count]
        
        y = np.array(y)
        
        for i, (color, label) in enumerate(zip(colors[:class_count], labels)):
            class_data = X[y == i+1]
            ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], c=color, label=label)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 25),
                                 np.linspace(y_min, y_max, 25),
                                 np.linspace(z_min, z_max, 25))
        
        grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        Z = classify_samples(grid, means, covs, priors)
        Z = Z.reshape(xx.shape)
        
        cmap = plt.cm.RdYlBu if class_count == 2 else plt.cm.viridis
        scatter = ax.scatter(xx, yy, zz, c=Z.ravel(), cmap=cmap, alpha=0.2, s=10)
        
        plt.colorbar(scatter, ax=ax, label='类别')
        
        ax.set_title(f"{'二' if class_count == 2 else '三'}类问题：三维特征的决策边界 (错误率: {error_rate:.2%})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.legend()
        plt.show()

    @staticmethod
    def plot_error_rates(error_rates):
        """
        绘制不同维度特征的错误率柱状图
        """
        plt.figure(figsize=(10, 6))
        plt.bar(['1D', '2D', '3D'], error_rates)
        plt.title("不同维度特征的错误率")
        plt.xlabel("特征维度")
        plt.ylabel("错误率")
        for i, v in enumerate(error_rates):
            plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')
        plt.show()

def run_classification(classes, dims, priors):
    """
    运行分类实验
    @param classes: 包含各个类别数据的列表
    @param dims: 要测试的特征维度列表
    @param priors: 各类别的先验概率
    """
    error_rates = []  # 错误率列表
    for dim in dims:
        X = np.vstack([cls[:, :dim] for cls in classes])  # 将所有类别的数据合并为一个数组  （根据dim取前dim个特征）
        true_labels = [i+1 for i, cls in enumerate(classes) for _ in range(len(cls))]  # 真实标签列表 10个1，10个2，10个3
        means = [estimate_parameters(cls[:, :dim])[0] for cls in classes]  # 计算每个类别的均值向量 （根据dim取前dim个特征）
        covs = [estimate_parameters(cls[:, :dim])[1] for cls in classes]  # 计算每个类别的协方差矩阵 （根据dim取前dim个特征）
        
        # 根据不同维度特征的分类器对样本进行分类
        predicted_labels = classify_samples(X, means, covs, priors)  # 对每个样本进行分类   返回数组中的每个元素是一个样本的类别标签
        error_rate = calculate_error_rate(true_labels, predicted_labels)  # 计算错误率
        error_rates.append(error_rate)  # 将错误率添加到列表中
        
        print(f"{'二' if len(classes) == 2 else '三'}类问题使用{dim}维特征的分类器错误率: {error_rate:.2%}")  # 打印错误率
        if dim <= 2:
            Visualizer.plot_decision_boundary(X, true_labels, means, covs, priors, error_rate, dim)  # 绘制决策边界
        elif dim == 3:
            Visualizer.plot_decision_boundary_3d(X, true_labels, means, covs, priors, error_rate, len(classes))  # 绘制三维决策边界
    
    Visualizer.plot_error_rates(error_rates)  # 绘制错误率柱状图

def main():
    # 二类问题
    run_classification([class_1, class_2], [1, 2, 3], [0.5, 0.5])  # 传入两类数据，3个维度特征，先验概率为0.5
    
    # 三类问题
    run_classification([class_1, class_2, class_3], [1, 2, 3], [1/3, 1/3, 1/3]) # 传入三类数据，3个维度特征，先验概率为1/3

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    main()