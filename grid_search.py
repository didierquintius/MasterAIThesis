# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:44:19 2020

@author: didie
"""
time_steps = [50]
trials = [500]
pred_archs = [[10]]
node_CNN = [2]
node_Dense = [5]
kernel_size = [[3]]
strides = [[2]]
learning_rate = [1e-3]
from Runv2 import runModel
from tqdm import tqdm
import pandas as pd

time_step_trials = [[time, trial] for time in time_steps for trial in trials]
nodes = [[node1, node2] for node1 in node_CNN for node2 in node_Dense]
clas_archs = [[node, kernel, stride, lr] for node in nodes for kernel in kernel_size for stride in strides for lr in learning_rate]
len_data, len_pred, len_clas=(len(time_step_trials), len(pred_archs), len(clas_archs))
longest = max(len_data, len_pred, len_clas)
result = pd.DataFrame([], columns = ["mean_train_pred", "std_train_pred", "train_performance_clas", "STOP_pred", "STOP_clas", "area_accuracy", "true_positive", "true_negative", "mean_mse", "std_mse"])
for i in tqdm(range(longest)):
    time_step, trial  = time_step_trials[i % len_data]
    pred_arch = pred_archs[i % len_pred]
    nodes, kernel, stride, lr = clas_archs[i % len_clas]
    output = runModel(0.9, 0.9, 5, 10, time_step, trial, pred_arch, nodes, kernel, stride, lr)
    result.loc[i] = output



