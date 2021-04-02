# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:43:57 2021

@author: didie
"""
from OptimalGridSearchResults import findOptimalvalues
from generate_centers import calculate_centers
from BrainAreaFunctions import train_brain_area
import numpy as np
from tqdm import tqdm
import pickle

def trainMLModel(results):
    optimal_hyper_params = findOptimalvalues(results)
    centers = np.unique(results['brain_area']).tolist()
    _, dipole_centers = calculate_centers(centers = centers)
    n_dipoles = 1000
    NNs = {}
    CNNs = {}
    for dipole in tqdm(range(n_dipoles)):
        params = optimal_hyper_params[centers[dipole_centers[dipole]]]
        mse_pred, mse_clas, truepositive_clas, truenegative_clas, STOP, STOP_clas, NN, CNN = train_brain_area(dipole, params, add_networks=True)
        NNs[dipole] = NN
        CNNs[dipole] = CNN   
        
    pickle.dump((NNs, CNNs), open('.Networks.pkl','wb'))