# -*- coding: utf-8 -*-
"""
Created on Sat May 30 03:12:29 2020

@author: didie
"""
import pandas as pd
elec_list = pd.read_excel("../Data/Electrode_position.xlsx", sheet_name = "Sheet2", header = None)
elec_groups = pd.read_excel("../Data/Electrode_position.xlsx", sheet_name = "Sheet3", header = None)
#%%
test = np.ones((108, 50))

for i, elec_group in enumerate(elec_groups.columns):
    for j, elec in enumerate(elec_list.transpose().values[0]):
        test[j, i] = elec in elec_groups.iloc[:, elec_group].transpose().values
        #%%
import pickle

pickle.dump(test, open("mask_matrix.pkl", "wb"))