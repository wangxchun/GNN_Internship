# -*- coding: utf-8 -*-

import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
'''
a = pd.DataFrame(pd.read_csv('result0.3.txt',sep='  ',header=None))
a = a.values
print(a[0])
print(a[50005000])
'''
def build_karate_club_graph():  # 这个是生成DGL库接受的图结构的函数
    # scr和dst分别代表图结构的流出节点和流入节点
    a = pd.DataFrame(pd.read_csv('result0.30.txt',sep='\t',header=None))
    print(a[0])

    src = a.iloc[:, [0]]
    dst = a.iloc[:, [1]]
    src = np.array(src.T).tolist()[0]
    dst = np.array(dst.T).tolist()[0]
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # DGL图创建成功
    return dgl.DGLGraph((u, v))

G = build_karate_club_graph()  # 调用上面写的函数生成DGL图
G = dgl.add_self_loop(G)
#通过将其转换为networkx图来可视化该图
#nx_G = G.to_networkx().to_undirected()
#pos = nx.kamada_kawai_layout(nx_G)
#nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
#plt.show()
#plt.savefig('graph.png')

import torch
import torch.nn as nn
import torch.nn.functional as F

embed = nn.Embedding(10000, 128)
# node_feats = torch.tensor(pd.read_excel('feature.xlsx',header=None).values)
# G.ndata['feat'] = node_feats
inputs = embed.weight
#embed = nn.Embedding(52157, 6)
#G.ndata['feat'] = embed.weight

from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):  # 构建GCN模型
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)  # 第一层图卷积
        self.conv2 = GraphConv(hidden_size, num_classes)  # 第二层图卷积

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)  # 第一层图卷积
        h = torch.relu(h)
        h = self.conv2(g, h)  # 第二层图卷积
        return h


net = GCN(128, 96, 16)#设置两个卷积层的输出维度
# inputs = node_feats#传入node_feature
# #labeled_nodes表示有类别标签的节点；labels代表class
# labeled_nodes = torch.tensor([0, 1, 3, 4, 5, 9, 13, 12, 25])  # torch中图节点从0号节点开始
# labels = torch.tensor([3, 2, 1, 6, 0, 5, 0, 4, 7])  # torch中Class从0开始计

# import itertools
logits = net(G, inputs)
# print(logits)
sofM=logits.detach().numpy()
df = pd.DataFrame(sofM)
writer = pd.ExcelWriter(r'result-gcn-16-0.30-.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
