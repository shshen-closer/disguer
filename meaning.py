import numpy as np
import pandas as pd
import time, datetime
from tqdm import tqdm
from sklearn.model_selection import KFold,StratifiedKFold
import json,os
from sklearn.model_selection import train_test_split
import sys

jiedian = np.load("data/jiedian.npy", allow_pickle=True)
list_file = os.listdir('results/')

aaa = []
for i in list_file:
    fff = np.load('results/' + str(i))
    aaa.append(fff)
print(np.shape(aaa))
pred_labels = np.mean(aaa, axis = 0)
start = 0
lens = 0
finalres = []
for lll in jiedian:
    pres = pred_labels[start:start + lll]
    start += lll
    finalres.append(",".join([str(p) for p in pres]))
    lens+=len(pres)
print(lens)


df_submit = pd.DataFrame({"responses":finalres})
df_submit.to_csv("prediction.csv",index=False)
