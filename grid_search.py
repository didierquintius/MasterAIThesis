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
import pickle
from Visualize_functions import plot_results
import time
def grid_search_params(run, time_steps = [50], trials = [10000],
                        pred_nodes = [[100]],
                        nodes_CNN = [19],
                        nodes_Dense = [20],
                        kernel_size = [[8]],  
                        strides = [[3 ]],
                        preportion_pred = [5],
                        preportion_clas = [9],
                        learning_rates = [1e-4],
                        batch_sizes_pred = [100],
                        batch_sizes_clas = [10],
                        val_tresholds = [1e-8]):
    random.seed(run)
    clas_nodes =[[node_CNN, node_Dense] for node_CNN in nodes_CNN for node_Dense in nodes_Dense]
    params = []
    for var  in [time_steps,trials, pred_nodes,clas_nodes,kernel_size, strides,preportion_pred, 
                         preportion_clas, learning_rates, learning_rates, batch_sizes_pred, 
                         batch_sizes_clas,val_tresholds, val_tresholds]:
        params += [var[int(random.random() * len(var))]]
    
    return tuple(params)

result = pd.DataFrame([], columns = ["time_steps", 'trials', 'pred_arch', 'nodes_CNN', 'nodes_Dense', 'kernel',
                                     'stride',  'lr_pred', 'lr_clas', 'preportion_pred', 'preportion_clas',
                                      'batch_pred', 'batch_clas', 'val_treshold_pred',
                                     'val_treshold_clas',"mean_train_pred", 
                                     "std_train_pred", "mean_train_clas",
                                     "std_train_clas", "STOP_pred", "STOP_clas",
                                     "area_accuracy", "true_positive", "true_negative", 
                                     "mean_mse", "std_mse", "time"])

#%%
prev_data_pred = {}
prev_data_clas = {}
runs = 1
repeats = 1
for i in tqdm(range(0,1)):
    params = grid_search_params(i)
    start = time.time()
    output, NeuralNets, STOP_pred, CNN_Nets, STOP_clas, train_pred, train_clas, optimizers_pred, optimizers_clas = runModel(0.9, 0.9, 50, 100, params, seed = 0)

    # for seed in tqdm(range(1,repeats)):
    #     print(output[21], output[22], output[24])
    #     for area in range(len(NeuralNets)):
    #         prev_data_pred[area] = (NeuralNets[area], STOP_pred[area], train_pred[area], optimizers_pred[area]) 
    #         prev_data_clas[area] = (CNN_Nets[area], STOP_clas[area], train_clas[area], optimizers_clas[area])
    #     output, NeuralNets, STOP_pred, CNN_Nets, STOP_clas, train_pred, train_clas, optimizers_pred, optimizers_pred = runModel(0.9, 0.9, 50, 100, params,False,prev_data_pred,prev_data_clas, seed)
    
    result.loc[i] = output + [time.time() - start]
    pickle.dump(result, open("result.pkl", "wb"))
#%%
plot_results(result)