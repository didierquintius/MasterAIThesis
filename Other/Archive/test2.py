# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:37:39 2021

@author: Quintius
"""
beta_values = {}
import numpy as np
from scipy import stats
#%%

def updateBetaValues(beta_values):
    for param, param_values in beta_values.items():
        
        top_param_value = min(param_values, key = lambda x: param_values[x][0] / param_values[x][1])
        alpha, beta = param_values[top_param_value]
        top_dist_values = np.random.beta(alpha, beta, alpha + beta)
        for param_value, values in param_values.items():
            alpha, beta = values
            dist_values = np.random.beta(alpha, beta, alpha + beta)
            pvalue = stats.ttest_ind(top_dist_values, dist_values)[1]
            if pvalue < 0.05:
                del beta_values[param][param_value]
    
    return beta_values                
        