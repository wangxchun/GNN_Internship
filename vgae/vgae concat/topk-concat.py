# -*- coding: utf-8 -*-

import numpy as np
import sys
import pickle as pkl
import networkx as nx 
import scipy.sparse as sp
import os
#os.chdir(r'C:\Users\Administrator\Desktop\python_workspace\8-24欧式距离')
import pandas as pd
from tqdm import tqdm


adj1 = pd.read_excel("result-gcn-16-0.25-.xlsx")
adj1 = adj1.drop(adj1.columns[0], axis=1)
adj2 = pd.read_excel("result-gcn-16-0.30-.xlsx")
adj2 = adj2.drop(adj2.columns[0], axis=1)
adj3 = pd.read_excel("result-gcn-16-0.35-.xlsx")
adj3 = adj3.drop(adj3.columns[0], axis=1)
adj4 = pd.read_excel("result-gcn-16-0.40-.xlsx")
adj4 = adj4.drop(adj4.columns[0], axis=1)
adj = pd.concat([adj1, adj2, adj3, adj4], axis=1)
adj = j = adj.values
print(adj)
print(adj.shape)
topk = 51

from scipy.spatial.distance import pdist
df = pd.DataFrame(columns=['l1', 'l2', 'l3','l4', 'l5', 'l6','l7', 'l8', 'l9','l10',
'l11', 'l12','l13', 'l14', 'l15','l16', 'l17', 'l18','l19', 'l20', 
'l21', 'l22','l23', 'l24', 'l25','l26', 'l27', 'l28','l29', 'l30',
'l31', 'l32','l33', 'l34', 'l35','l36', 'l37', 'l38','l39', 'l40',
'l41', 'l42','l43', 'l44', 'l45','l46', 'l47', 'l48','l49', 'l50',
'l51'])
flag = False
for i in range(10000):
    c = []
    for j in range(10000):
        X = np.vstack([adj[i],adj[j]])
        d2 = pdist(X,'euclidean')
        # print(d2)
        c.append(d2[0])
        # print(j)
    # inds = np.argsort(c)[-topk:]#最大值
    if(i%100==0):
        print(i)
    # c = c
    inds = np.argsort(c)[:topk]#最小值
    # print(inds)
    # inds = map(c.index, heapq.nsmallest(3, c))
    df.loc[i] = inds

writer = pd.ExcelWriter('topk-gcn-concat.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
# print(a[1][1])
# print(a)
# output = open('data-2.xls','w',encoding='gbk')
# output.write('s1\t1\n')
# for i in range(len(a)):
# 	print(i)
# 	for j in range(len(a[i])):
# 		output.write(str(a[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
# 		output.write('\t')   #相当于Tab一下，换一个单元格
# 	output.write('\n')       #写完一行立马换行
# output.close()

# a = a.numpy()
# for i in enumerate(len(a)):
#     print("i")
    # print("j")
# print(a)
# df = pd.DataFrame(a)
# writer = pd.ExcelWriter('result_eucl.xlsx')
# df.to_excel(writer, 'sheet_1', float_format='%.2f')
# writer.save()
# writer.close()
