# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:38:37 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
import torch
import math
import os
import pickle
from tqdm import tqdm
# os.chdir(r'E:\python\代码\python workspace1\mnistGNN-master')#邻接矩阵所在文件夹

#b = np.array([[0,1,1,1,1],[1,0,1,0,0],[1,1,0,1,0],[1,0,1,0,0],[1,0,0,0,0]])
#a = pd.DataFrame(b)

with open('traj_lcss_distance', 'rb') as df:
    dis_lcss = pickle.load(df)
df.close()
dis_lcss = pd.DataFrame(dis_lcss)

b = dis_lcss < 0.30
a = b.astype(int)

n = 10000#n = 节点个数
x, c = [], []#储存边关系
for i in tqdm(range(0,n)):
    for j in range(i+1,n):
        src = a.iloc[j,i]#定位两个节点之间是否存在连边关系
        # print(src)
        if src >= 1:
            x.append(i)
            c.append(j)

d = np.array([x,c])
# d = torch.tensor([x,c])
print(d.T)
df = pd.DataFrame(d.T)
df.to_csv('result0.30.txt',sep='\t',index=False,header=None)
#writer = pd.ExcelWriter(r'result-0830.xlsx')
#df.to_excel(writer, 'sheet_1', float_format='%.2f')
#writer.save()
#writer.close()
