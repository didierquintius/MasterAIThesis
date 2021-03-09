# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:57:22 2021

@author: didie
"""
from BrainAreaFunctions import train_brain_area
import pandas as pd
from tqdm import tqdm
import pickle
#%%
params = dict(nodes_pred = 50,
                nodes_Conv_clas = 45,
                nodes_Dense_clas = 19,
                kernel_size = 7, 
                strides = 1,
                learning_rate_pred = 1e-5,
                learning_rate_clas = 1e-4,
                batch_sizes_pred = 25,
                batch_sizes_clas = 5,
                val_treshold_pred = 1e-8,
                val_treshold_clas = 1e-8,
                max_val_amount_pred = 50,
                max_val_amount_clas = 50,
                val_freq_pred = 100,
                val_freq_clas = 5,
                EPOCHS_pred = 30,
                EPOCHS_clas = 20,
                trials = 1000,
                time_steps = 100,
                brain_areas = 1000)
#%%
resultaten = pickle.load(open('./eerste_resultaten.pkl','rb'))
for brain_area in tqdm([89]):
    mse_pred, mse_clas, truepositive_clas, truenegative_clas, STOP, STOP_clas = train_brain_area(brain_area, params, plot = True)
    resultaten.loc[brain_area] = [mse_pred, mse_clas, truepositive_clas, truenegative_clas, STOP, STOP_clas]
    pickle.dump(resultaten, open('./eerste_resultaten.pkl','wb'))