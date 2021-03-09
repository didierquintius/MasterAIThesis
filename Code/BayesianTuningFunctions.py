# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:11:32 2021

@author: didie
"""
import seaborn as sns
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats import weightstats as stests
import numpy as np
import re

def plotBetaValues(beta_values, brain_area, comb = [[]], save = True, results_folder = ''):
        for param, param_values in beta_values.items():
            dist_values =pd.DataFrame([])
            for param_value, alpha_beta in param_values.items():
                alpha, beta = alpha_beta
                dist_values[param_value] = np.random.beta(alpha, beta, 10000)
            
            dist_values = dist_values.melt()
            plot = sns.displot(dist_values, x="value", hue="variable", kind="kde", fill=True, palette = sns.color_palette("hls", len(param_values)))
            plot.set(title = param)
            if save: plt.savefig(results_folder + './Plots/' + param + '_' + str(comb[0][0]) + '_' + str(comb[1][0]) + '_' + str(comb[2][0]) + '_' + str(brain_area) + ".png")

def updateBetaValues(beta_values):
    
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


def update_beta_values(output1, output2, param_values, run, beta_values, runs):
    pred_regex = re.compile('.+_pred')
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

