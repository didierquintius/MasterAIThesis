# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:41:25 2021

@author: didie
"""

from itertools import combinations_with_replacement
import numpy as np
import torch, pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
#%%
results = pickle.load(open("C:/Users/didie/Documents/MasterAIThesis/Code/Results-17-3/TrainResults.pkl",'rb'))
#%%
elecs_xy = np.genfromtxt("C:/Users/didie/DATA/MasterAIThesis/Positions_electrodes.csv",
                   delimiter=';')
loc_elecs = {'x':elecs_xy[:,1].astype('int'), 'y': elecs_xy[:, 2].astype('int')}
#%%
X = np.random.randint(low= 0,high = 2, size = (100000,108)) *  0.9 + 0.001
X_torch = torch.Tensor(X)
#%%
dipoles = 1000
elecs = 108
elec_map = np.zeros((15, 9, dipoles))
for dipole in tqdm(range(dipoles)):
    nn = results['NNs'][dipole]
    y = nn(X_torch).detach().numpy()
    model = LinearRegression().fit(X, y)
    for elec in range(elecs):
        elec_map[loc_elecs['x'][elec], loc_elecs['y'][elec], dipole] = np.abs(model.coef_[0,elec])
