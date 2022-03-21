# -*- coding: utf-8 -*-

## pca特征降维
# 导入相关模块
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
from numpy.linalg import eig
# from sklearn.datasets import load_iris
import pickle
# import numpy as np
import sys
# import pickle as pkl
import networkx as nx 
import scipy.sparse as sp
# import numpy as np
import pandas as pd

with open('traj_discret_frechet_distance', 'rb') as df:
    dis_lcss = pickle.load(df)
    #[[int(v) for v in line.split()] + [1] for line in dis_lcss]
df.close()
# 导入数据
# iris = load_iris() # 150*4的矩阵，4个特征，行是个数，列是特征
X = dis_lcss
print(type(X))
k = 48  #选取贡献最大的前2个特征作为主成分，根据实际情况灵活选取
# Standardize by remove average通过去除平均值进行标准化
X = X - X.mean(axis=0)

# Calculate covariance matrix:计算协方差矩阵：
X_cov = np.cov(X.T, ddof=0)

# Calculate  eigenvalues and eigenvectors of covariance matrix
# 计算协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = eig(X_cov)

# top k large eigenvectors选取前k个特征向量
klarge_index = eigenvalues.argsort()[-k:][::-1]
k_eigenvectors = eigenvectors[klarge_index]

# X和k个特征向量进行点乘
X_pca = np.dot(X, k_eigenvectors.T)
df = pd.DataFrame(X_pca)
writer = pd.ExcelWriter('result_pca.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
# adj = X_pca
# topk = 51
# from scipy.spatial.distance import pdist
# df = pd.DataFrame(columns=['l1', 'l2', 'l3','l4', 'l5', 'l6','l7', 'l8', 'l9','l10',
# 'l11', 'l12','l13', 'l14', 'l15','l16', 'l17', 'l18','l19', 'l20', 
# 'l21', 'l22','l23', 'l24', 'l25','l26', 'l27', 'l28','l29', 'l30',
# 'l31', 'l32','l33', 'l34', 'l35','l36', 'l37', 'l38','l39', 'l40',
# 'l41', 'l42','l43', 'l44', 'l45','l46', 'l47', 'l48','l49', 'l50',
# 'l51'])
# for i in range(10000):
#     c = []
#     for j in range(10000):
#         X = np.vstack([adj[i],adj[j]])
#         d2 = pdist(X,'euclidean')
#         # print(d2)
#         c.append(d2[0])
#         # print(j)
#     # inds = np.argsort(c)[-topk:]#最大值
#     print(i)
#     # c = c
#     inds = np.argsort(c)[:topk]#最小值
#     # print(inds)
#     # inds = map(c.index, heapq.nsmallest(3, c))
#     df.loc[i] = inds
