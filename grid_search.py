# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:44:19 2020

@author: didie
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:44:19 2020
@author: didie
"""


#%%
import os
import random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from Runv2 import runModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from Visualize_functions import plot_results, plot_line
import time
def grid_search_params(run, time_steps = [50, 100, 200], trials = [1000],
                        pred_nodes = [[100]],
                        nodes_CNN = [19],
                        nodes_Dense = [20],
                        kernel_size = [[8]],  
                        strides = [[3]],
                        preportion_pred = [5],
                        preportion_clas = [9],
                        learning_rates = [1e-4],
                        batch_sizes_pred = [100],
                        batch_sizes_clas = [10],
                        val_tresholds = [1e-8],
                        brain_areas = [10, 16, 20],
                        max_val_amount = [50, 100, 200],
                        val_freq_pred = [20, 50, 100],
                        val_freq_clas = [1, 5, 10]):
    random.seed(run)
    clas_nodes =[[node_CNN, node_Dense] for node_CNN in nodes_CNN for node_Dense in nodes_Dense]
    params = []
    for var  in [time_steps,trials, pred_nodes,clas_nodes,kernel_size, strides,preportion_pred, 
                         preportion_clas, learning_rates, learning_rates, batch_sizes_pred, 
                         batch_sizes_clas,val_tresholds, val_tresholds, brain_areas,
                         max_val_amount, val_freq_pred, val_freq_clas]:
        params += [var[int(random.random() * len(var))]]
    
    return tuple(params)

result = pd.DataFrame([], columns = ["time_steps", 'trials', 'pred_arch', 'nodes_CNN', 'nodes_Dense', 'kernel',
                                     'stride',  'lr_pred', 'lr_clas', 'preportion_pred', 'preportion_clas',
                                      'batch_pred', 'batch_clas', 'val_treshold_pred',
                                     'val_treshold_clas', 'brain_areas',"max_val_amount", 'val_freq_pred', 
                                     'val_freq_clas',"mean_train_pred", 
                                     "std_train_pred", "mean_train_clas",
                                     "std_train_clas", "STOP_pred", "STOP_clas",
                                     "area_accuracy", "true_positive", "true_negative", 
                                     "mean_mse", "std_mse", "time"])

#%%
prev_data_pred = {}
prev_data_clas = {}
runs = 5 
for i in tqdm(range(runs)):
    params = grid_search_params(i)
    start = time.time()
    output, NeuralNets, val_losses = runModel(0.9, 0.9, int(0.5 * params[14]), params[14], params, seed = 0)    
    result.loc[i] = output + [time.time() - start]
    pickle.dump(result, open("result.pkl", "wb"))
#%%
#v1, v2 = val_losses
#plot_line(list(v1.values()), np.arange(10).tolist())
#plot_line(list(v2.values()), np.arange(10).tolist())