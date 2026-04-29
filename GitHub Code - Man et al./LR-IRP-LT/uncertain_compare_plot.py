import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置Seaborn主题
sns.set_style("whitegrid")
palette = ["black", "red", "blue"]  # 黑-红-蓝 配色

# 数据
uncertainty_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
# uncertainty_levels_00=[0.01, 0.03, 0.05, 0.07, 0.09]
methods = ["BFP", "RFP", "TSRFP"]
markers = ["o", "s", "D"]  # 圆点, 方块, 菱形

# BFP 数据
bfp_lr = [0.0044, 0.0074, 0.0236, 0.0052, 0.0062]
bfp_stock_out = [45993.144, 38502.566, 30378.682, 42075.678, 51916.624]
bfp_regret = [63, 56, 368, 48, 88]

# RFP 数据
rfp_lr = [0.0044, 0.0074, 0.0072, 0.0052, 0.0064]
rfp_stock_out = [46474.896, 38526.55, 31269.382, 41057.666, 52319.47]
rfp_regret = [63, 56, 43, 48, 94]

# TSRFP 数据
tsrfp_lr = [0.0044, 0.0072, 0.0066, 0.005, 0.005]
tsrfp_stock_out = [46311.668, 37998.314, 28309.148, 39519.842, 49915.922]
tsrfp_regret = [63, 52, 31, 43, 58]

# total_LR=np.array([[0.0044, 0.0044, 0.0044],
#        [0.0074, 0.0074, 0.0072],
#        [0.0236, 0.0072, 0.0066],
#        [0.0052, 0.0052, 0.005 ],
#        [0.0062, 0.0064, 0.005 ]])
#
# stock_out=np.array([[45993.144, 38502.566, 30378.682, 42075.678, 51916.624],
#                     [46474.896, 38526.55, 31269.382, 41057.666, 52319.47],
#                     [46311.668, 37998.314, 28309.148, 39519.842, 49915.922]]).T
#
# regret=np.array([[63, 56, 368, 48, 88],
# [63, 56, 43, 48, 94],
# [63, 52, 31, 43, 58]]).T


total_LR = np.array([
    [0.0074, 0.0082, 0.0062],
    [0.0052, 0.0052, 0.0046],
    [0.0040, 0.0042, 0.0036],
    [0.0034, 0.0036, 0.0034],
    [0.0056, 0.0056, 0.0054]
])


stock_out = np.array([
    [35741.614, 36148.254, 33191.626],
    [49033.068, 48473.978, 47913.290],
    [46877.604, 48864.374, 44827.894],
    [53001.674, 52063.512, 52958.562],
    [45789.378, 45424.822, 42541.278]
])

regret = np.array([
    [37, 43, 25],
    [30, 30, 20],
    [37, 40, 30],
    [29, 33, 29],
    [34, 34, 31]
])

# 画图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 三个子图

titles = ["Total LR", "Final Stock Out", "Regret"]
y_labels = ["Total LR", "Stock Out", "Regret (%)"]
data_sets = [total_LR, stock_out, regret]

# 绘制每个折线图
for i, (ax, title, y_label, data) in enumerate(zip(axes, titles, y_labels, data_sets)):
    for j, method in enumerate(methods):
        ax.plot(uncertainty_levels, data[:, j],
                marker=markers[j], markersize=8, linestyle="-",
                color=palette[j], label=method, linewidth=2)

    ax.set_title(title + " vs Uncertainty Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Uncertainty Level", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(title="Model", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 美化坐标轴
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# 调整布局
plt.tight_layout()
plt.show()
