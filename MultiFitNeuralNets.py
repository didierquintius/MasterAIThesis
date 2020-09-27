# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:10:28 2020

@author: didie
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from FitNeuralNet import fit_NeuralNets
from loadAndSplitMulti import loadAndSplitMulti
#%%
X, y, active_electrodes, no_brain_sources = loadAndSplitMulti(0.9, 0.9, 51, 100, time_steps = 100,  trials = 10000)
#%%

NeuralNets = fit_NeuralNets(X, y, no_brain_sources)  

