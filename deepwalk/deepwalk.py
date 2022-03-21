# -*- coding: utf-8 -*-

import networkx as nx
# create_using=nx.DiGraph() 表示构造的是有向图
G = nx.read_edgelist('result0.80.txt')#, create_using=nx.DiGraph())
#print(G.node())
print("finish loading data")

from scipy.special import logsumexp
import random

# 从 start_node 开始随机游走
def deepwalk_walk(walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

# 产生随机游走序列
def _simulate_walks(nodes, num_walks, walk_length):
    walks = []
    i = 0
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(walk_length=walk_length, start_node=v))
        print(i)
        i += 1
    return walks

# 得到所有节点
nodes = list(G.nodes())
print(nodes)
# 得到序列
walks = _simulate_walks(nodes, num_walks=100, walk_length=20)

from gensim.models import Word2Vec
import pandas as pd
# 默认嵌入到100维

w2v_model = Word2Vec(walks,size=64,sg=1,hs=1)
# 打印其中一个节点的嵌入向量
print("finish loading data")
vacab = w2v_model.wv.vocab.keys()
print("finish loading data")
b = w2v_model.wv[vacab]
print("finish loading data")

df = pd.DataFrame(b)
#df = pd.DataFrame(vacab)
writer = pd.ExcelWriter('result-dw-0.8-.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
#vacab = w2v_model.wv.vocab.keys()
'''
df = pd.DataFrame(vacab)
writer = pd.ExcelWriter('data_result_vacab.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
'''
#bbb = pd.DataFrame()
#for key in vacab:
#    bb = w2v_model.most_similar(key, topn=10)
#    bbb = bbb.append(bb)
#    
#writer = pd.ExcelWriter('result_data.xlsx')
#bbb.to_excel(writer, 'sheet_1', float_format='%.2f')
#writer.save()
#writer.close()    

#b = pd.DataFrame.from_dict(vacab, orient='index', columns=['key'])

#b = pd.DataFrame(my_list).T
#b = w2v_model.most_similar(vacab[1], topn=10)
#print(vacab[1,1])
''' 
b = w2v_model.wv[vacab]
#输出节点表示向量
w2v_model.most_similar("张三丰", topn=10)
import pandas as pd

df = pd.DataFrame(b)
writer = pd.ExcelWriter('result_b.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
#print(b)

#输出节点名称
df = pd.DataFrame(vacab)
writer = pd.ExcelWriter('result_vacab.xlsx')
df.to_excel(writer, 'sheet_1', float_format='%.2f')
writer.save()
writer.close()
'''
