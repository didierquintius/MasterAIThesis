# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:44:19 2020

@author: didie
"""

time_steps = [40]
trials = [3000]
pred_nodes = [[i] for i in range(25, 201, 25)]
node_CNN = [i for i in range(10, 31, 3)]
node_Dense = [i for i in range(5, 41, 5)]
kernel_size = [[i] for i in range(2,11,2)]
strides = [[i] for i in range(1,4)]
preportions = [i for i in range(3, 11, 2)]
learning_rates = [i * 1e-8 for i in range(2,10, 2)]
batch_sizes_pred = [20, 50, 100]
batch_sizes_clas = [10, 20, 30]
val_tresholds = [1e-8]
runs = 500
#%%
import os
import random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from Runv2 import runModel
from Visualize_functions import plot_hyperparameters
from tqdm import tqdm
import pandas as pd

time_step_trials = [[time, trial] for time in time_steps for trial in trials]
nodes = [[node1, node2] for node1 in node_CNN for node2 in node_Dense]
pred_archs = [[node, preportion, learning_rate, batch_size, val_treshold] for node in pred_nodes 
              for preportion in preportions for learning_rate in learning_rates 
              for batch_size in batch_sizes_pred for val_treshold in val_tresholds]
clas_archs = [[node, kernel, stride, lr, preportion, batch_size, val_treshold] for node in nodes for kernel in kernel_size 
              for stride in strides for lr in learning_rates for preportion in preportions
              for batch_size in batch_sizes_pred for val_treshold in val_tresholds]
len_data, len_pred, len_clas=(len(time_step_trials), len(pred_archs), len(clas_archs))
longest = max(len_data, len_pred, len_clas)
result = pd.DataFrame([], columns = ["time_step", "trial", "pred_arch","preportion",
                                     'learning_rate_pred', 'batch_pred', 'val_treshold_pred',
                                     "nodes_CNN",'nodes_Dense', "kernel", "stride", 
                                     "lr", "preportion_clas",'batch_clas', 'val_treshold_clas',"mean_train_pred", 
                                     "std_train_pred", "mean_train_clas",
                                     "std_train_clas", "STOP_pred", "STOP_clas",
                                     "area_accuracy", "true_positive", "true_negative", 
                                     "mean_mse", "std_mse"])

#%%
for i in tqdm(range(runs)):
    random.seed(i)
    time_step, trial  = time_step_trials[int(random.random() * (len_data))]
    pred_arch, preportion, learning_rate_pred, batch_pred, val_treshold_pred = pred_archs[int(random.random() * (len_pred))]
    nodes, kernel, stride, lr, preportion_clas, batch_clas, val_treshold_clas = clas_archs[int(random.random() * (len_clas))]
    output, NeuralNets, CNN_Nets = runModel(0.9, 0.9, 25, 50, time_step, trial, pred_arch, learning_rate_pred, preportion, batch_pred, val_treshold_pred,
                      nodes, kernel, stride, lr, False, preportion_clas, batch_clas, val_treshold_clas)
    result.loc[i] = [time_step, trial, pred_arch[0], preportion, learning_rate_pred, batch_pred, val_treshold_pred, nodes[0], nodes[1],  kernel[0], stride[0], lr, preportion_clas, batch_clas, val_treshold_clas] + output
#%%
plot_hyperparameters(result, ['mean_train_pred', 'std_train_pred'], ["pred_arch","preportion",'learning_rate_pred', 'batch_pred', 'val_treshold_pred'])
plot_hyperparameters(result, ['mean_train_clas', 'std_train_clas'], ["nodes_CNN",'nodes_Dense', "kernel", "stride", "lr", "preportion_clas",'batch_clas', 'val_treshold_clas'])
