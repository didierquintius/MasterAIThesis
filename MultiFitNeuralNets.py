# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:10:28 2020

@author: didie
"""
import os, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from FitNeuralNet import fit_NeuralNets
from loadAndSplitMulti import loadAndSplitMulti
#%%
snr, cnr, noisy, neurons = (0.9, 0.9, 5, 20)
X, y, active_electrodes, no_brain_sources = loadAndSplitMulti(snr, cnr, noisy, neurons, time_steps = 20,  trials = 100)
#%%

NeuralNets = fit_NeuralNets(X, y, no_brain_sources)
#%%
file = "../../Data/NeuralNets"+ str(snr) +"_"+ str(cnr) +"_"+ str(noisy) +"_"+ str(neurons) +".pkl"
pickle.dump(NeuralNets, open(file, "wb"))