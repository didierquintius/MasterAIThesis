# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:41:40 2020

@author: didie
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
from eLoreta import eloreta
from tqdm import tqdm
import numpy as np

def performELORETA(EEG_data, labels, source, sources, reduction = 74, gamma = 0.01):
    
    projection_matrix = pickle.load(open("../Data/projection_matrix.pkl", "rb"))
    projection_matrix = projection_matrix[:, range(0,projection_matrix.shape[1], reduction)]
    
    P = eloreta(projection_matrix, gamma)
    elecs, time, loops = EEG_data.shape
    H = np.zeros((3 ,time, loops))
    used_sources = (labels.T @ np.arange(len(sources))).astype(int)
    
    for l in tqdm(range(loops)):
        result = EEG_data[:, :, l].T @ P[:, sources[used_sources[l]]]
        H[:, :, l] = (result/ np.linalg.norm(result, axis = 0)).T
    
    return H