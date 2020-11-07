# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:28 2020

@author: didie
"""
import pickle, torch

def load_and_split_data(validation_perc = 0.1, test_perc = 0.1):
    EEG_data, labels, source = pickle.load(open("../Data/EEG/data_1_1_200.pkl", "rb"))
    
    elecs, time_steps, trials = EEG_data.shape
    
    train_ind = int((1 - validation_perc - test_perc) * trials)
    val_ind = int((1 - test_perc) * trials)
    indexes = {"train": range(train_ind), "val": range(train_ind, val_ind),
               "test": range(val_ind, trials)}
    
    XandY  = {"X": {},"y": {}}
    for data in XandY.values():
        for data_type, index in indexes.items():
            data[data_type] = torch.Tensor(EEG_data[:,:,index].copy()).transpose(0,1).reshape(len(index) * time_steps,elecs)
    
    return XandY["X"], XandY["y"]