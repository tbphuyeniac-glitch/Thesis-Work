import matplotlib.pyplot as plt
import numpy as np

# 数据准备
product_types = ['Cheap & Small', 'Cheap & Medium', 'Cheap & Large',
                 'Normal & Small', 'Normal & Medium', 'Normal & Large',
                 'Expensive & Small', 'Expensive & Medium', 'Expensive & Large']

# BFP 数据
bfp_lr = [0.18, 0.028, 0.062, 0.0016, 0.0066, 0.0096, 0.00049, 0.0045, 0.0036]
bfp_cost = [11317, 6894, 2933, 11048, 4758, 1522, 13634, 5060, 1901]
bfp_stock_out = [5725, 6372, 6556, 42627, 48250, 45595, 176442, 200026, 247359]
bfp_stock_out_rate = [0.226, 0.248, 0.238, 0.242, 0.252, 0.270, 0.22, 0.226, 0.256]

# RFP 数据
rfp_lr = [0.131, 0.0304, 0.0645, 0.00168, 0.00721, 0.00955, 0.000492, 0.00450, 0.00360]
rfp_cost = [8146, 5510, 2787, 10765, 3789, 1508, 11459, 4948, 1886]
rfp_stock_out = [5296, 6441, 6397, 42916, 47405, 45594, 175907, 205074, 247359]
rfp_stock_out_rate = [0.214, 0.254, 0.234, 0.244, 0.248, 0.27, 0.218, 0.228, 0.256]

# TSRFP 数据
tsrfp_lr = [0.01590, 0.0243, 0.0686, 0.00145, 0.00591, 0.00959, 0.00049, 0.00436, 0.00361]
tsrfp_cost = [12645, 6604, 3456, 10536, 4326, 1588, 13647, 5063, 2891]
tsrfp_stock_out = [5391, 5984, 6211, 39432, 48139, 45976, 166033, 199528, 235987]
tsrfp_stock_out_rate = [0.21, 0.236, 0.234, 0.226, 0.254, 0.272, 0.208, 0.226, 0.246]

# 创建子图
fig, axs = plt.subplots(2, figsize=(14, 12))
# fig.suptitle('Comparison among BFP, RFP, and TSRFP Models', fontsize=16)

# # 绘制 Total LR
axs[0].plot(product_types, bfp_lr, 'bo-', label='BFP')
axs[0].plot(product_types, rfp_lr, 'r^--', label='RFP')
axs[0].plot(product_types, tsrfp_lr, 'ks-.', label='TSRFP')
axs[0].set_ylabel('Total LR')
axs[0].set_xticklabels(product_types, rotation=45, ha='right', size=10)
axs[0].legend()

# 绘制 Total Cost
# axs[0].plot(product_types, bfp_cost, 'bo-', label='BFP')
# axs[0].plot(product_types, rfp_cost, 'r^--', label='RFP')
# axs[0].plot(product_types, tsrfp_cost, 'ks-.', label='TSRFP')
# axs[0].set_ylabel('Total Cost')
# axs[0].set_xticklabels(product_types, rotation=15, ha='right', size=10)
# axs[0].legend()

# # 绘制 Final Stock-out
axs[1].plot(product_types, bfp_stock_out, 'bo-', label='BFP')
axs[1].plot(product_types, rfp_stock_out, 'r^--', label='RFP')
axs[1].plot(product_types, tsrfp_stock_out, 'ks-.', label='TSRFP')
axs[1].set_ylabel('Final Stock-out')
axs[1].set_xticklabels(product_types, rotation=45, ha='right', size=10)
axs[1].legend()

# 绘制 Stock-out Rate
# axs[1].plot(product_types, bfp_stock_out_rate, 'bo-', label='BFP')
# axs[1].plot(product_types, rfp_stock_out_rate, 'r^--', label='RFP')
# axs[1].plot(product_types, tsrfp_stock_out_rate, 'ks-.', label='TSRFP')
# axs[1].set_ylabel('Stock-out Rate')
# axs[1].set_xticklabels(product_types, rotation=15, ha='right', size=10)
# axs[1].legend()

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# 显示图表
plt.show()