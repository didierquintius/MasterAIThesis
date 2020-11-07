# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:28 2020

@author: didie
"""
import pickle, torch, os, random
import numpy as np
from itertools import compress
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%%
def splitData(EEG_data, sources, activity, brain_areas, train_perc = 0.7,
                      val_perc = 0.1):  
    
    elecs, time_steps, trials = EEG_data.shape
        
    XandY  = {"y": {}, "X": {}}
    original_data = {"X": EEG_data, "y" : activity}
    no_cols = {"X" : elecs, "y" : 1}
    
    for XorY, data in XandY.items():
        for brain_area in range(brain_areas):
            data[str(brain_area)]= {}
            relevant_trials = [brain_area in source for source in sources]
            relevant_data = original_data[XorY][:, :, relevant_trials]
            total_trials = range(sum(relevant_trials))
            random.seed(0)
            train = random.sample(total_trials, int(train_perc * len(total_trials)),)
            remaining_trials = list(set(total_trials) -set(train))
            val = random.sample(remaining_trials, int(val_perc * len(total_trials)),)
            test = list(set(total_trials) - set(val))

            indexes = {"train": train, "val":  val,  "test": test}
          
            if XorY == "y":
                relevant_sources = list(compress(sources, relevant_trials))
                relevant = [[neuron == brain_area for neuron in source] for source in relevant_sources]
                relevant_data = relevant_data.swapaxes(1,0)
                relevant_data = np.sum(relevant_data * np.array(relevant).T, axis = 1)
            
                for partition_name, index in indexes.items():
                    data[str(brain_area)][partition_name] = torch.Tensor(relevant_data[:, index].copy().reshape(no_cols[XorY], time_steps * len(index), order = "F"))
                    data[str(brain_area)][partition_name] = data[str(brain_area)][partition_name].transpose(0,1)

            else:
                for partition_name, index in indexes.items():
                    data[str(brain_area)][partition_name] = torch.Tensor(relevant_data[:,:,index].copy().reshape(no_cols[XorY], time_steps * len(index), order = "F"))
                    data[str(brain_area)][partition_name] = data[str(brain_area)][partition_name].transpose(0,1)
      
    return XandY["X"], XandY["y"]