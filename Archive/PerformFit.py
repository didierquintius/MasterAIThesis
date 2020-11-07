# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:10:28 2020

@author: didie
"""
import os
import numpy as np
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork import NeuralNet
from load_and_split_data import load_and_split_data
from FitNeuralNet import FitNeuralNet
from MathematicalInverse import performELORETA
#%%
X, y, EEG_data, elecs, source = load_and_split_data(0.9, 0.9, 40)
#%%
architectures = [[25]]
learningrates = [5e-4]
results = pd.DataFrame([], columns = ["architecture", "learning_rate","test_performance", "sd_test_performance"])

ind = 0
temp_results = []
repetitions = 1
for architecture in architectures:
    for lr in learningrates:        
        for i in range(repetitions):        
            Net = NeuralNet(X["train"].shape[1], architecture)
            result = FitNeuralNet(Net, X, y, learning_rate = lr, max_val_amount= 25,
                                       EPOCHS = 5, val_freq = 10)
            temp_results += [result["test_performance"]]
        results.loc[ind] = [architecture, lr, np.mean(temp_results), np.std(temp_results)]
        ind += 1
        temp_results = []
        print(ind)
