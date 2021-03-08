# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:41:00 2021

@author: Quintius
"""

import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
import re, os, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from statsmodels.stats import weightstats as stests
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from BrainAreaFunctions import train_brain_area

def updateBetaValues(beta_values):
    plotBetaValues(beta_values, save = False)
    for param, param_values in beta_values.items():
        
        top_param_value = min(param_values, key = lambda x: param_values[x][0] / param_values[x][1])
        alpha, beta = param_values[top_param_value]
        np.random.seed(0)
        top_dist_values = np.random.beta(alpha, beta, int(alpha + beta))
        iter_values = copy(list(param_values.values()))
        param_value_names = list(param_values.keys())

        for i, values in enumerate(iter_values):
            alpha, beta = values
            np.random.seed(0)
            dist_values = np.random.beta(alpha, beta, int(alpha + beta))
            pvalue = stests.ztest(top_dist_values, dist_values, value=0,alternative='two-sided')[1]
            if pvalue < 1e-10:
                del beta_values[param][param_value_names[i]]
    plotBetaValues(beta_values, save = False)
    return beta_values   

def generate_prob(param, value, beta_values):
    alpha, beta = beta_values[param][value]
    return np.random.beta(alpha, beta)

def generate_params(beta_values):
    chosen_param_values = {}
    for param, param_values in beta_values.items():
        param_values = list(param_values.keys())
        probs = [generate_prob(param, value, beta_values) for value in param_values]
        chosen_param_values[param] = param_values[np.argmin(probs)]
    return chosen_param_values


def update_beta_values(output1, output2, param_values, run, beta_values):
    for param, value in param_values.items():
        if bool(re.match(pred_regex, param)): output = output1
        else: output = output2
        alpha, beta = beta_values[param][value]
        if output > 0.5: output= 0.5
        beta_values[param][value] = (alpha + output * (1 * run / runs), beta + 1 - output * (1 + run / runs))
    return beta_values

def initiateBetaValues(params):
    beta_values = {}
    for param, param_values in params.items():
        beta_values[param] = {}
        for value in param_values:
            beta_values[param][value] = (np.finfo('float').tiny, 1)
    return beta_values

def plotBetaValues(beta_values, comb = [[]], save = True):
    for param, param_values in beta_values.items():
        dist_values =pd.DataFrame([])
        for param_value, alpha_beta in param_values.items():
            alpha, beta = alpha_beta
            dist_values[param_value] = np.random.beta(alpha, beta, 10000)
        
        dist_values = dist_values.melt()
        plot = sns.displot(dist_values, x="value", hue="variable", kind="kde", fill=True, palette = sns.color_palette("hls", len(param_values)))
        plot.set(title = param)
        if save: plt.savefig('./Plots/' + param + '_' + str(comb[0][0]) + '_' + str(comb[1][0]) + '_' + str(comb[2][0]) + '_' + str(brain_area) + ".png")

#%%
# list all  fixed hyper parameters with their possible values
fixed_params = dict(time_steps = [100],
                trials = [100],
                brain_areas = [100])
# list all other hyper parameters
params = dict(nodes_pred = np.arange(50,200,25).tolist(),
                nodes_Conv_clas = np.arange(10,50, 5),
                nodes_Dense_clas = np.arange(5,30,2),
                kernel_size = np.arange(2,10), 
                strides = np.arange(1,4),
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
                EPOCHS_clas = np.arange(20, 40, 2).tolist())#,
                #CNN_structure = ["1", "n"],
                #Smoothing = ['True', 'False'],
                #TrainWithNoisySources_pred = ['True', 'False'],
                #val_perc = [0.1, 0.2, 0.05],
                #test_trials = [50, 100, 200])
#%%
runs = 500
fixed_comb = [[]]
pred_regex = re.compile('.+_pred')

# generate all possible combinations of fixed hyper parameters  
for param, param_values in fixed_params.items():
    fixed_comb = [comb + [[param_value]] for param_value in param_values for comb in fixed_comb]
output_values = ["mse_pred", "mse_clas", "true_positive_clas", 'true_negative_clas', "STOP_pred", "STOP_clas", "time", 'brain_area']
results = pd.DataFrame([], columns = list(params.keys()) + list(fixed_params.keys()) + output_values)
#%%
for brain_area in [0]:
    for comb in fixed_comb:
        params['time_steps'], params['trials'], params['brain_areas'] = tuple(comb)
          
        # generate starting alpha and beta parameters for each value
        beta_values = initiateBetaValues(params)
      
        for run in tqdm(range(runs)):
            param_values = generate_params(beta_values)
            start = time()
            output = train_brain_area(brain_area, param_values)
            results.loc[len(results)] = list(param_values.values()) + list(output) + [time() - start] + [brain_area]
            beta_values = update_beta_values(output[0], 1 - output[2], param_values, run, beta_values)
            pickle.dump(results, open('./results.pkl', 'wb'))
            pickle.dump(beta_values, open('./beta'+ str(brain_area) +'.pkl', 'wb'))
            if (run > 250) & ((run % 50) == 0):
                beta_values = updateBetaValues(beta_values)
        
        plotBetaValues(beta_values, comb)
            
    
