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

def grid_search_params(run, time_steps = [50], trials = [8000],
                        pred_nodes = [[i] for i in range(25, 201, 25)],
                        nodes_CNN = [i for i in range(10, 31, 3)],
                        nodes_Dense = [i for i in range(5, 41, 5)],
                        kernel_size = [[i] for i in range(2,11,2)],
                        strides = [[i] for i in range(1,4)],
                        preportions = [i for i in range(3, 11, 2)],
                        learning_rates = [i * 1e-4 for i in range(2,10, 2)],
                        batch_sizes_pred = [50, 100, 200],
                        batch_sizes_clas = [10, 20, 30],
                        val_tresholds = [1e-8]):
    random.seed(run)
    clas_nodes =[[node_CNN, node_Dense] for node_CNN in nodes_CNN for node_Dense in nodes_Dense]
    params = []
    for var  in [time_steps,trials, pred_nodes,clas_nodes,kernel_size, strides,preportions, 
                         preportions, learning_rates, learning_rates, batch_sizes_pred, 
                         batch_sizes_clas,val_tresholds, val_tresholds]:
        params += [var[int(random.random() * len(var))]]
    
    return tuple(params)

result = pd.DataFrame([], columns = ["time_steps", 'trials', 'pred_arch', 'nodes_CNN', 'nodes_Dense', 'kernel',
                                     'stride', 'preportion_pred', 'preportion_clas', 'lr_pred',
                                     'lr_clas', 'batch_pred', 'batch_clas', 'val_treshold_pred',
                                     'val_treshold_clas',"mean_train_pred", 
                                     "std_train_pred", "mean_train_clas",
                                     "std_train_clas", "STOP_pred", "STOP_clas",
                                     "area_accuracy", "true_positive", "true_negative", 
                                     "mean_mse", "std_mse"])

#%%
prev_data_pred = {}
prev_data_clas = {}
runs = 150
repeats = 1
for i in tqdm(range(50,runs)):
    params = grid_search_params(i)
    output, NeuralNets, STOP_pred, CNN_Nets, STOP_clas, train_pred, train_clas = runModel(0.9, 0.9, 25, 50, params, seed = 0)

    # for seed in tqdm(range(1,repeats)):
    #     print(output[19], output[20])
    #     for area in range(len(NeuralNets)):
    #         prev_data_pred[area] = (NeuralNets[area], STOP_pred[area], train_pred)
    #         prev_data_clas[area] = (CNN_Nets[area], STOP_clas[area], train_clas)
    #     output, NeuralNets, STOP_pred, CNN_Nets, STOP_clas, train_pred, train_clas = runModel(0.9, 0.9, 5, 10, params,False,prev_data_pred,prev_data_clas, seed)
    
    result.loc[i] = output    
    
