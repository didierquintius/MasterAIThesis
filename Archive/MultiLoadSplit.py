# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:28 2020

@author: didie
"""
import pickle, torch, os
import numpy as np
from itertools import compress
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
def Multi_load_and_split(snr, cnr, noise_sources, parts, validation_perc = 0.1,
                        test_perc = 0.1, train_ind = 100):
    EEG_data, sources, activity = pickle.load(open("../Data/EEG/data_" + str(snr) + 
                                                "_" + str(cnr) + "_" + str(noise_sources) + ".pkl", "rb"))
    
    elecs, time_steps, trials = EEG_data.shape
    train_ind = train_ind * time_steps
    
    EEG_data = EEG_data.reshape((elecs, time_steps * trials), order = "F")
    activity = activity.reshape((3, time_steps * trials), order = "F")
        
    XandY  = {"y": {}, "X": {}}
    original_data = {"X": EEG_data, "y" : activity}
    no_cols = {"X" : elecs, "y" : 1}
    
    for XorY, data in XandY.items():
        for part in range(parts):
            data[str(part)]= {}
            relevant_trials = [part in source for source in sources]
            relevant_trials_total = np.repeat(relevant_trials, time_steps)                   
            relevant_data = original_data[XorY][:, relevant_trials_total]
            
            indexes = {"train": range(train_ind), 
                       "val":  range(train_ind, 
                                     (train_ind + int((sum(relevant_trials_total) - train_ind) / 2))),
                       "test": range(train_ind + int((sum(relevant_trials_total) - train_ind )/ 2),
                                     sum(relevant_trials_total))}
          
            if XorY == "y":
                relevant_sources = list(compress(sources, relevant_trials))
                relevant = [[neuron == part for neuron in source] for source in relevant_sources]
                relevant = np.repeat(relevant, time_steps, axis = 0)
                relevant_data = np.sum(relevant_data * relevant.T, axis = 0)  
                for partition_name, index in indexes.items():
                    data[str(part)][partition_name] = torch.Tensor(relevant_data[index].copy())
                    data[str(part)][partition_name] = data[str(part)][partition_name].view(no_cols[XorY], len(index)).transpose(0,1)

            else:
                for partition_name, index in indexes.items():
                    data[str(part)][partition_name] = torch.Tensor(relevant_data[:,index].copy())
                    data[str(part)][partition_name] = data[str(part)][partition_name].view(no_cols[XorY], len(index)).transpose(0,1)
      
    return XandY["X"], XandY["y"], EEG_data, sources, activity