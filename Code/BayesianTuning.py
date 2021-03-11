# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:41:00 2021

@author: Quintius
"""

import numpy as np
import pandas as pd
from datetime import date
import re, os, pickle
from tqdm import tqdm
from time import time
from BayesianTuningFunctions import *
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from BrainAreaFunctions import train_brain_area


#%%
# list all  fixed hyper parameters with their possible values
fixed_params = dict(time_steps = [10],
                trials = [10],
                brain_areas = [1000])
# list all other hyper parameters
params = dict(nodes_pred = np.arange(50,200,25).tolist(),
                nodes_Conv_clas = np.arange(10,50, 5).tolist(),
                nodes_Dense_clas = np.arange(5,30,2).tolist(),
                kernel_size = np.arange(2,10).tolist(), 
                strides = np.arange(1,4).tolist(),
                learning_rate_pred = [5e-3,1e-4,5e-5, 1e-5,5e-6, 1e-6],
                learning_rate_clas = [5e-3,1e-4,5e-5, 1e-5,5e-6, 1e-6],
                batch_sizes_pred = [25, 50, 100, 200],
                batch_sizes_clas = [5, 10, 25, 50],
                val_treshold_pred = [1e-8, 1e-9, 1e-10],
                val_treshold_clas = [1e-8, 1e-9, 1e-10],
                max_val_amount_pred = [50, 100, 200],
                max_val_amount_clas = [50, 100, 200],
                val_freq_pred = [25, 50, 100],
                val_freq_clas = [2, 5, 10],
                EPOCHS_pred = np.arange(20, 40, 2).tolist(),
                EPOCHS_clas = np.arange(20, 40, 2).tolist())
#%%
runs = 500
fixed_comb = [[]]


# generate all possible combinations of fixed hyper parameters  
for param, param_values in fixed_params.items():
    fixed_comb = [comb + [[param_value]] for param_value in param_values for comb in fixed_comb]
output_values = ["mse_pred", "mse_clas", "true_positive_clas", 'true_negative_clas', "STOP_pred", "STOP_clas", "time", 'brain_area']
results = pd.DataFrame([], columns = list(params.keys()) + list(fixed_params.keys()) + output_values)
#%%
for brain_area in [0]:
    for comb in fixed_comb:
        results_folder = './results_' + str(date.today().day) + str(date.today().month) + str(int(time()))
        if not os.path.isdir(results_folder): os.mkdir(results_folder)
        params['time_steps'], params['trials'], params['brain_areas'] = tuple(comb)
          
        # generate starting alpha and beta parameters for each value
        beta_values = initiateBetaValues(params)
      
        for run in tqdm(range(runs)):
            param_values = generate_params(beta_values)
            start = time()
            output = train_brain_area(brain_area, param_values)
            results.loc[len(results)] = list(param_values.values()) + list(output) + [time() - start] + [brain_area]
            beta_values = update_beta_values(output[0], 1 - output[2], param_values, run, beta_values, runs)
            pickle.dump(results, open(results_folder + './results.pkl', 'wb'))
            pickle.dump(beta_values, open(results_folder + './beta'+ str(brain_area) +'.pkl', 'wb'))
            if (run > 250) & ((run % 50) == 0):
                beta_values = updateBetaValues(beta_values)
        
        plotBetaValues(beta_values, brain_area, comb, results_folder = results_folder)
            
    
