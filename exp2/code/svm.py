import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 原始数据
w3_data = np.array([
    [-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2],
    [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]
])

w4_data = np.array([
    [-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
    [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]
])

def transform_features(X):
    """将二维特征转换为高维特征：[1, x1, x2, x1², x1x2, x2²]"""
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([
        np.ones_like(x1),  # 1
        x1,                # x1
        x2,                # x2
        x1**2,             # x1²
        x1*x2,             # x1x2
        x2**2              # x2²
    ])

def prepare_data(n_samples):
    """准备训练数据"""
    X_w3 = w3_data[:n_samples]
    X_w4 = w4_data[:n_samples]
    X = np.vstack([X_w3, X_w4])
    y = np.array([1]*len(X_w3) + [-1]*len(X_w4))
    return X, y

def train_and_analyze_svm(n_samples):
    """在6维空间训练SVM并分析结果"""
    # 准备数据
    X, y = prepare_data(n_samples)
    X_transformed = transform_features(X)
    # 训练SVM
    svm = SVC(kernel='linear', C=1000)
    svm.fit(X_transformed, y)
    
    return get_svm_results(svm, X, X_transformed, y)

def train_and_analyze_svm_original(n_samples):
    """在原始2维空间训练SVM并分析结果"""
    # 准备数据
    X, y = prepare_data(n_samples)
    # 训练SVM
    svm = SVC(kernel='linear', C=5000)
    svm.fit(X, y)
    
    return get_svm_results(svm, X, X, y)

def get_svm_results(svm, X, X_train, y):
    """获取SVM训练结果"""
    w = svm.coef_[0]
    b = svm.intercept_[0]
    margin = 2 / np.linalg.norm(w)          # 最小间隔
    support_vectors = X[svm.support_]       # 支持向量
    
    # 计算所有样本到超平面的距离
    predictions = svm.predict(X_train)
    decision_values = svm.decision_function(X_train)
    
    # 检查是否严格线性可分（所有样本都在正确的一侧，且距离超平面至少有一定距离）
    tolerance = 1e-3  # 设置一个小的容忍度
    is_strictly_separable = np.all(decision_values * y >= tolerance)
    
    # 计算分类准确率
    accuracy = np.mean(predictions == y)
    
    return {
        'w': w,
        'b': b,
        'margin': margin,
        'support_vectors': support_vectors,
        'is_separable': is_strictly_separable,
        'accuracy': accuracy,
        'decision_values': decision_values
    }

def print_results(result, space_type="6维"):
    """打印SVM结果"""
    if space_type == "2维":
        print(f"超平面方程：{result['w'][0]:.4f}x1 + {result['w'][1]:.4f}x2 + ({result['b']:.4f}) = 0")
    else:
        print(f"超平面方程：{result['w'][0]:.4f}·1 + {result['w'][1]:.4f}x1 + {result['w'][2]:.4f}x2 + "
              f"{result['w'][3]:.4f}x1² + {result['w'][4]:.4f}x1x2 + {result['w'][5]:.4f}x2² + ({result['b']:.4f}) = 0")
    print(f"间隔：{result['margin']:.4f}")
    print("支持向量：")
    for sv in result['support_vectors']:
        print(f"({sv[0]:.2f}, {sv[1]:.2f})")

def plot_svm_results(X, y, result, title, is_transformed=False):
    """可视化SVM结果"""
    plt.figure(figsize=(10, 8))
    
    # 创建网格来绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # 准备网格点
    if is_transformed:
        grid_points = transform_features(np.c_[xx.ravel(), yy.ravel()])
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 计算决策函数
    Z = np.dot(grid_points, result['w']) + result['b']
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和间隔边界
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    
    # 绘制样本点
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='r', marker='o', label='w3', s=100)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='b', marker='s', label='w4', s=100)
    
    # 标记支持向量
    plt.scatter(result['support_vectors'][:, 0], result['support_vectors'][:, 1],
                s=300, facecolors='none', edgecolors='g', label='支持向量')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title, fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.axis('equal')

def visualize_results(n_samples):
    """可视化指定样本数量的结果"""
    X, y = prepare_data(n_samples)
    
    # 训练并可视化原始2维空间的结果
    result_orig = train_and_analyze_svm_original(n_samples)
    plot_svm_results(X, y, result_orig, 
                    f'原始2维空间 (样本数={n_samples})\n间隔: {result_orig["margin"]:.4f}')
    
    # 训练并可视化6维空间的结果（投影到2维）
    result_transformed = train_and_analyze_svm(n_samples)
    plot_svm_results(X, y, result_transformed, 
                    f'6维空间投影到2维 (样本数={n_samples})\n间隔: {result_transformed["margin"]:.4f}',
                    is_transformed=True)

def print_separability_analysis(n_samples):
    """分析样本的可分性"""
    X, y = prepare_data(n_samples)
    
    # 2维空间分析
    result_orig = train_and_analyze_svm_original(n_samples)
    print(f"\n原始2维空间（{n_samples}个样本）：")
    print(f"分类准确率: {result_orig['accuracy']:.4f}")
    print(f"最小间隔: {np.min(np.abs(result_orig['decision_values'])):.4f}")
    print(f"严格线性可分: {'是' if result_orig['is_separable'] else '否'}")
    
    # 6维空间分析
    result_transformed = train_and_analyze_svm(n_samples)
    print(f"\n6维空间（{n_samples}个样本）：")
    print(f"分类准确率: {result_transformed['accuracy']:.4f}")
    print(f"最小间隔: {np.min(np.abs(result_transformed['decision_values'])):.4f}")
    print(f"严格线性可分: {'是' if result_transformed['is_separable'] else '否'}")

def analyze_separability(svm, X, y, space_type="2维"):
    """详细分析线性可分性"""
    decision_values = svm.decision_function(X)
    margins = decision_values * y
    
    print(f"\n{space_type}空间分离性分析：")
    print("各样本到决策边界的距离（带符号）：")
    for i, (x, margin) in enumerate(zip(X, margins)):
        label = "ω₃" if y[i] == 1 else "ω₄"
        print(f"样本{i+1} ({label}): ({x[0]:.2f}, {x[1]:.2f}) -> 距离: {margin:.4f}")
    
    min_margin = np.min(margins)
    print(f"\n最小间隔: {min_margin:.4f}")
    print(f"是否严格线性可分: {'是' if min_margin >= 1e-3 else '否'}")

def main():
    """主函数"""
    # 问题(a)的结果
    print("\n=== 问题(a)的结果 ===")
    print("\n6维空间：")
    result_a = train_and_analyze_svm(1)
    print_results(result_a)
    
    print("\n原始2维空间：")
    result_a_orig = train_and_analyze_svm_original(1)
    print_results(result_a_orig, "2维")

    # 问题(b)的结果
    print("\n=== 问题(b)的结果 ===")
    print("\n6维空间：")
    result_b = train_and_analyze_svm(2)
    print_results(result_b)
    
    print("\n原始2维空间：")
    result_b_orig = train_and_analyze_svm_original(2)
    print_results(result_b_orig, "2维")

    # 问题(c)的结果
    print("\n=== 问题(c)的结果 ===")
    for i in [1, 2, 3, 5, 10]:  # 选择几个关键的样本数量进行分析
        print(f"\n--- 使用前{i}个样本的分析 ---")
        print_separability_analysis(i)

    # 使用所有样本的训练结果
    print("\n=== 使用所有样本的训练结果 ===")
    print("\n原始2维空间（全部样本）：")
    result_full_orig = train_and_analyze_svm_original(10)
    print_results(result_full_orig, "2维")
    
    print("\n映射到6维空间后（全部样本）：")
    result_full = train_and_analyze_svm(10)
    print_results(result_full)

    # 线性可分性结果
    print(f"\n原始2维空间是否线性可分：{'是' if result_full_orig['is_separable'] else '否'}")
    print(f"6维空间是否线性可分：{'是' if result_full['is_separable'] else '否'}")

    # 添加可视化部分
    print("\n=== 生成可视化结果 ===")
    
    # 可视化问题(a)的结果（1个样本）
    visualize_results(1)
    
    # 可视化问题(b)的结果（2个样本）
    visualize_results(2)
    
    # 可视化所有样本的结果
    visualize_results(10)
    
    plt.show()

if __name__ == "__main__":
    main()