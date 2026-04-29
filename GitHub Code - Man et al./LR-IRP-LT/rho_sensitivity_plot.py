# Re-import necessary libraries after execution state reset
import matplotlib.pyplot as plt
import numpy as np


# # Data
# rho = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
# intervals = ["[0,0.25]", "[0.25,0.5]", "[0.5,0.75]", "[0.75,1]", "TOTAL [0,1]"]
#
# stock_out = np.array([
#     [0.248, 0.248, 0.248, 0.248, 0.248],
#     [0.248, 0.248, 0.248, 0.248, 0.248],
#     [0.248, 0.248, 0.248, 0.248, 0.248],
#     [0.248, 0.248, 0.248, 0.248, 0.248],
#     [0.248, 0.248, 0.248, 0.248, 0.248],
#     [0.248, 0.248, 0.248, 0.248, 0.248],
#     [0.246, 0.246, 0.246, 0.246, 0.248],
#     [0.25, 0.25, 0.25, 0.25, 0.252],
#     [0.244, 0.244, 0.244, 0.244, 0.248],
#     [0.248, 0.248, 0.248, 0.248, 0.25],
#     [0.252, 0.252, 0.252, 0.252, 0.252]
# ])
#
# total_cost = np.array([
#     [3747.598] * 5, [3747.598] * 5, [3747.598] * 5, [3747.598] * 5, [3747.598] * 5,
#     [3747.598] * 5, [3738.082] * 5, [3730.984] * 5, [2673.338] * 5, [1918.552] * 5, [1675.366] * 5
# ])
#
# # total_LR = np.array([
# #     [0.006012] * 5, [0.006012] * 5, [0.006012] * 5, [0.006012] * 5, [0.006012] * 5,
# #     [0.006012] * 5, [0.005998] * 5, [0.00599] * 5, [0.006304] * 5, [0.006742] * 5, [0.024072] * 5
# # ])
#
# colors = ["black", "red", "blue", "purple", "green"]
#
# # Plot function
# def plot_metric(rho, metric, title, ylabel):
#     plt.figure(figsize=(8, 5))
#     for i in range(len(intervals)):
#         plt.plot(rho, metric[:, i], marker="o", linestyle="-", color=colors[i], label=intervals[i])
#     plt.xlabel("rho")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     # plt.show()
#
# # Plot stock out
# plot_metric(rho, stock_out, "Stock out rate vs rho", "Stock out rate")
#
# # Plot total cost
# plot_metric(rho, total_cost, "Total cost vs rho", "Total cost")

# Plot total LR
# plot_metric(rho, total_LR, "Total LR vs rho", "Total LR")

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 设定 rho 作为横坐标（假设有 10 个数据点）
rho = np.linspace(-1, 1, 11)

# 数据（示例数据，替换成你的实际数据） TSRFP
stock_out_stracture = [0.252] * 6 + [0.252, 0.252, 0.254, 0.254, 0.254]
stock_out_total = [0.252] * 6 + [0.252, 0.252, 0.254, 0.254, 0.254]

total_cost_stracture = [4709.77] * 6 + [4710.53, 4710.53, 4704.092, 4704.092, 4704.092]
total_cost_total = [4709.77] * 6 + [4710.53, 4712.212, 4704.092,4704.092, 4704.486]

#   RO
# stock_out_stracture = [0.248] * 6 + [0.246, 0.25,  0.244, 0.248,0.252]
# stock_out_total = [0.248] * 7 + [0.252, 0.248, 0.25, 0.252]
#
# total_cost_stracture = [3747.598] * 6 + [3738.082, 3730.984,2673.338,1918.552,1675.366]
# total_cost_total = [3747.598] * 6 + [3738.082,3730.984,2673.338,1918.552,1675.366]

# total_LR_stracture = [0.005204] * 6 + [0.005204, 0.005204, 0.005132, 0.005132]
# total_LR_total = [0.005204] * 6 + [0.005204, 0.005204, 0.005132, 0.005132]

# 颜色和标记样式
colors = ['black', 'red']
markers = ['o', 's']

# 画图
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

# stock out
axes[0].plot(rho, stock_out_stracture, linestyle='-', marker=markers[0], color=colors[0], label='Structural uncertain')
axes[0].plot(rho, stock_out_total, linestyle='--', marker=markers[1], color=colors[1], label='Total uncertain')
axes[0].set_ylabel('Stock-out rate')
axes[0].set_xlabel('ρ')
axes[0].legend()
axes[0].grid(True)

# total cost
axes[1].plot(rho, total_cost_stracture, linestyle='-', marker=markers[0], color=colors[0], label='Structural uncertain')
axes[1].plot(rho, total_cost_total, linestyle='--', marker=markers[1], color=colors[1], label='Total uncertain')
axes[1].set_ylabel('Total Cost')
axes[1].set_xlabel('ρ')
axes[1].legend()
axes[1].grid(True)

# total LR
# axes[2].plot(rho, total_LR_stracture, linestyle='-', marker=markers[0], color=colors[0], label='stracture[0,1]')
# axes[2].plot(rho, total_LR_total, linestyle='--', marker=markers[1], color=colors[1], label='TOTAL [0,1]')
# axes[2].set_ylabel('Total LR')
# axes[2].set_xlabel('ρ')
# axes[2].legend()
# axes[2].grid(True)

plt.tight_layout()
plt.show()
