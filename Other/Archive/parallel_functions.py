# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:59:52 2020

@author: Quintius
"""
import random
#%%
def combos(list1, list2):
    all_combinations = [x + [y] for y in list2 for x in list1]
    return all_combinations

def make_combos(lists):
    all_combos = [[x] for x in lists[0]]
    for list_ in lists[1:]:
        all_combos = combos(all_combos, list_)
    return all_combos
        
#%%

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
    
    clas_nodes =[[node_CNN, node_Dense] for node_CNN in nodes_CNN for node_Dense in nodes_Dense]
    params = []
    for var  in [time_steps,trials, pred_nodes,clas_nodes,kernel_size, strides,preportions, 
                         preportions, learning_rates, learning_rates, batch_sizes_pred, 
                         batch_sizes_clas,val_tresholds, val_tresholds]:
        params += [var[int(random.random() * len(var))]]
    
    return tuple(params)