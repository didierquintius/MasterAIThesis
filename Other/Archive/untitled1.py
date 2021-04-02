# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:00:05 2021

@author: didie
"""
import pickle, os, random
import numpy as np

def close_active_dipoles(trials, n_relevant_dipoles = 1000):
    
    distances = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\dipole_distances.pkl', 'rb'))
    close_dipoles = []
    for dipole in range(1000):
        close_dipoles += [np.where(distances[dipole,:] < 20)[0].tolist()]
    source_dipoles = []
    for trial in range(trials):
        source_dipoles += [random.sample(close_dipoles[trials % n_relevant_dipoles], 3)]
    return source_dipoles