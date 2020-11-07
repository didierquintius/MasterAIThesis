# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:10:28 2020

@author: didie
"""
import os
from time import time
from tqdm import tqdm
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork import NeuralNet
from FitNeuralNet import FitNeuralNet
from loadAndSplitMulti import loadAndSplitMulti
#%%
X, y, EEG_data, elecs, source = loadAndSplitMulti(0.9, 0.9, 20, 50)
#%%
architecture = [[50], [10, 5], [15, 5]]
lr =  [1e-4]
grid_search = pd.DataFrame({"archictecture":{},
                            "learning rate":{},
                            "train":{},
                            "val":{},
                            "test":{},
                            "running_time":{}})
#%%
count = 0
part = 1
for a in architecture:
    for l in lr:
        count += 1
        start = time()
        Net = NeuralNet(X[str(part)]["train"].shape[1], a, output = 1)    
        result = FitNeuralNet(Net, X[str(part)], y[str(part)], learning_rate = l, max_val_amount= 40,
                               EPOCHS = 50, val_freq = 5)
        print(count, result["train_performance"],result["val_performance"],result["test_performance"])
        print()
        grid_search.loc[count] = [a,l,result["train_performance"],result["val_performance"],result["test_performance"], time() - start]
    

        

