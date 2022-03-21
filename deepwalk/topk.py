# -*- coding: utf-8 -*-

import numpy as np
import sys
import pickle as pkl
import networkx as nx 
import scipy.sparse as sp
import os
import pandas as pd
from tqdm import tqdm

#adj = np.array(pd.read_excel('result-dw-0.80.xlsx',header=None))
#adj = pd.read_excel("result_pca.xlsx").values
adj = pd.read_excel("result-gcn-0.30.xlsx")
adj = adj.drop(adj.columns[0], axis=1).values
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

writer = pd.ExcelWriter('topk-dw-0.80-.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()

