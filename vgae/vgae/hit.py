import pandas as pd
import numpy as np
from tqdm import tqdm, trange
row = 10000

def get_lines(file_path):
    with open(file_path, 'rb') as f:
        for line in f:
            yield line

topk_original = [[] for i in range(row)]
f = open('topk_original.txt',"r")
for i in range(row):
    line = f.readline().strip().strip('\n').strip('\r').split(" ")
    line = list(map(eval, line))
    for item in line:
        topk_original[i].append(item)
topk_original =  pd.DataFrame(topk_original).values
#print(topk_original)
    
topk_embedding = [[] for i in range(row)]
f = open('topk-gcn-0.30.txt',"r")
for i in range(row):
    line = f.readline().strip().strip('\n').strip('\r').split('\t')
    line = list(map(eval, line))
    #print(line)
    for item in line:
        topk_embedding[i].append(item)
topk_embedding =  pd.DataFrame(topk_embedding).values
#print(topk_embedding)
    
count = 0
for i in range(row):
    flag = True
    for item in topk_original[i]:
        if item not in topk_embedding[i]:
            flag = False
    if flag == True:
        count += 1
print(count/row)