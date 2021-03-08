# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:41:00 2021

@author: Quintius
"""
import numpy as np
import random 
import re
from test1 import train_brain_area
from tqdm import tqdm
# list all hyper parameters with their possible values
params = dict(time_steps = [50, 100, 200],
                trials = [1000, 5000, 10000],
                brain_areas = [200, 500, 1000],
                nodes_pred = np.arange(50,200,25).tolist(),
                nodes_Conv_clas = np.arange(10,50, 5),
                nodes_Dense_clas = np.arange(5,30,2),
                kernel_size = np.arange(2,10), 
                strides = np.arange(1,4),
                learning_rate_pred = [1e-4,5e-5, 1e-5,5e-6, 1e-6],
                learning_rate_clas = [1e-4,5e-5, 1e-5,5e-6, 1e-6],
                batch_sizes_pred = [25, 50, 100, 200],
                batch_sizes_clas = [5, 10, 25, 50],
                val_treshold_pred = [1e-8, 1e-9, 1e-10],
                val_treshold_clas = [1e-8, 1e-9, 1e-10],
                max_val_amount_pred = [50, 100, 200],
                max_val_amount_clas = [50, 100, 200],
                val_freq_pred = [20, 50, 100],
                val_freq_clas = [1, 5, 10],
                EPOCHS_pred = np.arange(10, 30, 2).tolist(),
                EPOCHS_clas = np.arange(10, 30, 2).tolist(),
                CNN_structure = ["1", "n"],
                Smoothing = ['True', 'False'],
                TrainWithNoisySources_pred = ['True', 'False'])
#%%
runs = 500

# generate starting alpha and beta parameters for each value
beta_values = {}
for param, param_values in params.items():
    beta_values[param] = {}
    for value in param_values:
        if (param == 'time_steps') | (param == 'trials') | (param == 'brain_areas'):
            beta_values[param][value] = (1, 1000 * runs)
        else:
            beta_values[param][value] = (np.finfo('float').tiny, 1)

def generate_prob(param, value, beta_values = beta_values):
    alpha, beta = beta_values[param][value]
    return np.random.beta(alpha, beta)

def generate_params(params = params):
    chosen_param_values = {}
    for param, param_values in params.items():
        probs = [generate_prob(param, value) for value in param_values]
        chosen_param_values[param] = param_values[np.argmin(probs)]
    return chosen_param_values

pred_regex = re.compile('.+_pred')
def update_beta_values(output1, output2, param_values, beta_values = beta_values, params = params):
    for param, value in param_values.items():
        if bool(re.match(pred_regex, param)): output = output1
        else: output = output2
        alpha, beta = beta_values[param][value]
        beta_values[param][value] = (alpha + output, beta + 1 - output)
    return beta_values
    
for run in tqdm(range(runs)):
    param_values = generate_params()    
    output1, output2 = train_brain_area(0, param_values)
    beta_values = update_beta_values(output1, output2, param_values)
    
#%%
import seaborn as sns
import pandas as pd
for param, param_values in beta_values.items():
    dist_values =pd.DataFrame([])
    for param_value, alpha_beta in param_values.items():
        alpha, beta = alpha_beta
        dist_values[param_value] = np.random.beta(alpha, beta, 10000)
    
    dist_values = dist_values.melt()
    sns.displot(dist_values, x="value", hue="variable", kind="kde", fill=True)
        


