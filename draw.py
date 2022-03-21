# 需导入要用到的库文件
import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库

x = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
y = [0.0646, 0.0650, 0.0765, 0.0902, 0.0808, 0.0789, 0.0712, 0.0754, 0.0692, 0.0587, 0.0574, 0.0461, 0.0461, 0.0464, 0.0342, 0.0219]
fig = plt.figure()
ax = plt.subplot()
plt.title('hit-threshold',fontsize=20,color='black')
plt.xlabel('threshold',fontsize=15,color='black')
plt.ylabel('hit',fontsize=15,color='black')
ax.scatter(x, y, alpha=0.5)  # 绘制散点图，面积随机
plt.savefig('hit-gcn')
plt.show()