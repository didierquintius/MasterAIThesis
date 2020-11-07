# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:16:25 2020

@author: didie
"""

import numpy as np
import pandas as pd

def create_transformation(format_electrodes_1d, format_electrodes_2d):
        
    
    n, m = format_electrodes_2d.shape
    transformation = np.zeros((n,m,108))
    print(n, m)
    for i in range(n):
        for j in range(m):
            if not isinstance(format_electrodes_2d[i,j], str):
                next
            else:
                transformation[i, j, :]= format_electrodes_1d[0] == format_electrodes_2d[i,j]  
    return transformation
#%%

def format_data(source, EEG_Data, TIMEFRAMES_CNN = 12):
    
    def data_split(data, TIMEFRAMES_CNN = TIMEFRAMES_CNN):
        x, y, timeframes, trials = data.shape
        split_size = timeframes - TIMEFRAMES_CNN + 1
        split_data = np.empty((split_size, 15, 9, TIMEFRAMES_CNN, trials))
        
        for i in range(split_size):
            split_data[i, :, :, :, :] = data[:,:,i:(i + TIMEFRAMES_CNN),:]
    
        return split_data
    
    format_electrodes_2d = np.array(pd.read_excel("../Data/Electrode_position.xlsx", sheet_name = "Sheet1", header = None))
    format_electrodes_1d = pd.read_excel("../Data/Electrode_position.xlsx", sheet_name = "Sheet2", header = None)
    
    transformation = create_transformation(format_electrodes_1d, format_electrodes_2d)
    channels, timeframes, trials = EEG_Data.shape
    
    EEG_transformed = transformation @ EEG_Data.reshape((channels, timeframes * trials))
    EEG_transformed = EEG_transformed.reshape((15, 9, timeframes, trials))        
    
    EEG_split = data_split(EEG_transformed)
    # reverse the data
    EEG_split = np.flip(EEG_split, 0)
    source = np.flip(source, 1)
    source = source[:, (TIMEFRAMES_CNN - 1):, :]
    
    return EEG_split, source


