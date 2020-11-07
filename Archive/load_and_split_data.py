# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:28 2020

@author: didie
"""
import pickle, torch

def load_and_split_data(snr, cnr, noise_sources, validation_perc = 0.1,
                        test_perc = 0.1, transform = None):
    EEG_data, labels, source = pickle.load(open("../Data/EEG/data_" + str(snr) + 
                                                "_" + str(cnr) + "_" + str(noise_sources) + ".pkl", "rb"))
    
    elecs, time_steps, trials = EEG_data.shape
    
    train_ind = int((1 - validation_perc - test_perc) * trials)
    val_ind = int((1 - test_perc) * trials)
    
    indexes = {"train": range(train_ind), "val": range(train_ind, val_ind),
               "test": range(val_ind, trials)}
    
    XandY  = {"X": {},"y": {}}
    original_data = {"X": EEG_data, "y" : source}
    no_cols = {"X" : elecs, "y" : 3}
    
    for XorY, data in XandY.items():
        for partition_name, index in indexes.items():
            data[partition_name] = torch.Tensor(original_data[XorY][:,:,index].copy())
            data[partition_name] = data[partition_name].view(no_cols[XorY], len(index) * time_steps).transpose(0,1)
    
    return XandY["X"], XandY["y"], EEG_data, labels, source