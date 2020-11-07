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
MultiNet = {}
for i in range(50):
    MultiNet[str(i)] = NeuralNet(X["train"].shape[1], [25])

#%%
result = FitNeuralNet(Net, X, y, learning_rate = 5e-4, max_val_amount= 25,
                           EPOCHS = 5, val_freq = 10)

