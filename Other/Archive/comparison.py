# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:44:37 2020

@author: Quintius
"""
from Runv2 import importData
from performance_function import test_performance, most_likely_sources
import numpy as np
from elor

snr, cnr, noisy_areas, brain_areas, time_steps, trials = (0.9, 0.9, 25, 50, 50, 10000)

EEG_data, sources, noisy_sources, activity =  importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials)

area_accuracy, true_positive, true_negative, mean_mse, std_mse, time_series = test_performance(EEG_data, sources, activity, brain_areas, NeuralNets, CNN_Nets, False)

stds = time_series.std(axis = 2)
stds = (stds - stds.min())/(stds.max() - stds.min())

source= np.zeros((trials, brain_areas))
for trial in range(trials):
    source[trial, sources[trial,:]] = 1

area_accuracy, true_positive, true_negative = most_likely_sources(stds, source)


