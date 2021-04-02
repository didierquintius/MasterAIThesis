# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:25:05 2021

@author: didie
"""
import seaborn as sns
import numpy as np
import pandas as pd
import pickle, os
import matplotlib.pyplot as plt
from itertools import combinations
#%%
# mse_pred = np.random.normal(size = 1000)
# mse_clas = np.random.normal(size = 1000)
# TA = np.random.normal(size = 1000)
# TN = np.random.normal(size = 1000)
# TI = np.random.normal(size = 1000)

# distance_to_scalp = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\distance_to_scalp.pkl', 'rb'))
# measures = {'mse_pred': mse_pred, 'mse_clas': mse_clas, 'TA': TA, 'TN': TN, 'TI': TI}
# # #%%
# for measure in measures.keys():
#     plt.figure()
#     plot = sns.boxplot(x = 'distance_label', y = measure, data = hi)
#     plot.set(title = measure)
    
#     #plt.savefig(results_folder + '/Plots/' + param + '_' + str(comb[0][0]) + '_' + str(comb[1][0]) + '_' + str(comb[2][0]) + '_' + str(brain_area) + ".png")
#  #%%

# for measure1, measure2 in combinations(measures.keys(), 2):
#     plt.figure()
#     plot = sns.scatterplot(x = measure1, y = measure2, data = hi)
#     plot.set(title = measure)
    
#     #plt.savefig(results_folder + '/Plots/' + param + '_' + str(comb[0][0]) + '_' + str(comb[1][0]) + '_' + str(comb[2][0]) + '_' + str(brain_area) + ".png")

def evaluateTrainResults(mse_pred, mse_clas, TA, TN, TI, accuracy):
    
    distance_to_scalp = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\distance_to_scalp.pkl', 'rb'))
    measures = {'mse_pred': list(mse_pred.values()), 'mse_clas': list(mse_clas.values()),
                'TA': list(TA.values()), 'TN': list(TN.values()), 'TI': list(TI.values()),
                'accuracy': list(accuracy.values())}
    results = pd.DataFrame(measures)
    results['distance_label'] = pd.cut(distance_to_scalp, bins = [-1, 10, 20, 30, 100], labels = ['very shallow', 'shallow', 'deep', 'very deep'])
    for measure, values in measures.items():
        print(measure, np.mean(values), sep = ": ")

        
    for measure in measures.keys():
        plt.figure()
        plot = sns.boxplot(x = 'distance_label', y = measure, data = results)
        plot.set(title = measure)
    
    for measure1, measure2 in combinations(['mse_pred', 'mse_clas', 'accuracy'], 2):
        plt.figure()
        plot = sns.scatterplot(x = measure1, y = measure2, data = results, hue = 'distance_label')
        plot.set(title = measure1 + ' - ' + measure2)
        
def evaluateResults(mse_pred, mse_clas, accuracy, source_dipoles, noisy_dipoles, source_accuracy):
    measures = pd.DataFrame([])
    measures['mse_pred'] = mse_pred.reshape(-1)
    measures['mse_clas'] = mse_clas.reshape(-1)
    measures['accuracy'] = accuracy.reshape(-1)
    measures['source_accuracy'] = np.repeat([source_accuracy], repeats = mse_pred.shape[0], axis = 0).reshape(-1)
    measures['state'] = 'idle'
    measures.loc[source_dipoles.reshape(-1) == 1, 'state'] = 'source'
    measures.loc[noisy_dipoles.reshape(-1) == 1, 'state'] = 'noisy'
    distance_to_scalp = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\distance_to_scalp.pkl', 'rb'))
    distance_label = np.array(pd.cut(distance_to_scalp, bins = [-1, 10, 20, 30, 100], labels = ['very shallow', 'shallow', 'deep', 'very deep']))
    measures['location'] = np.repeat(distance_label, mse_pred.shape[1])
    emeasures = ['mse_pred', 'mse_clas', 'accuracy']
    for measure in emeasures: print(measure, measures[measure].mean(), sep = ': ')
    print('source_accuracy: ', measures['source_accuracy'].mean())
    print(measures.groupby(['state']).mean())
    print(measures.groupby(['location']).mean())
    print(measures.groupby(['state', 'location']).mean())
        
    for measure in emeasures:
        plt.figure()
        plot = sns.boxplot(x = 'location', y = measure, data = measures)
        plot.set(title = measure)
        plt.figure()
        plot = sns.boxplot(x = 'state', y = measure, data = measures)
        plot.set(title = measure)
        
    for measure1, measure2 in combinations(emeasures, 2):
        plt.figure()
        plot = sns.scatterplot(x = measure1, y = measure2, data = measures)
        plot.set(title = measure1 + ' - ' + measure2 )

        
        
        
    