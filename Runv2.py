# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:14 2020

@author: didie
"""
import pickle, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from Fit_functions import fitProjectionModels, fitClassificationModels
from DataSimulation_functions import EEG_signal
from performance_function import test_performance


def runModel(snr, cnr, noisy_areas, brain_areas, time_steps, trials, pred_arch, preportion,
             nodes, kernel, stride, lr, plot):
    def importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials):
        data_file = "../Data/EEG/data_" + str(snr) + "_" + str(cnr) + "_" + str(noisy_areas) + "_" + str(brain_areas) + "_" + str(time_steps) + "_" + str(trials) + ".pkl"
        
        if os.path.exists(data_file):            
            EEG_data, sources, activity = pickle.load(open(data_file, "rb"))
        else:
            EEG_data, sources, noisy_sources, activity =  EEG_signal(time_steps, trials, brain_areas,
                                                      snr, cnr, noisy_areas)
        
        return EEG_data, sources, noisy_sources, activity
    #%%
    
    EEG_data, sources, noisy_sources, activity =  importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials)
    EEG_data_test,  sources_test, noisy_sources_test, activity_test, EEG_data_trainval,  sources_trainval, noisy_sources_trainval, activity_trainval = splitTestData(EEG_data, sources, noisy_sources, activity)
    del EEG_data, sources, activity
    
    #%%
    NeuralNets, mean_train_pred, std_train_pred, val_losses_pred, STOP_pred = fitProjectionModels(EEG_data_trainval, sources_trainval, noisy_sources_trainval, activity_trainval, brain_areas, architecture = pred_arch, preportion = preportion)
    
    #%%
    CNN_Nets, mean_train_clas, std_train_clas, val_losses_clas, STOP_clas = fitClassificationModels(EEG_data_trainval,sources_trainval,brain_areas,NeuralNets, nodes, kernel, stride, lr)
    area_accuracy, true_positive, true_negative, mean_mse, std_mse = test_performance(EEG_data_test, sources_test, activity_test, brain_areas, NeuralNets, CNN_Nets, plot)
    
    STOP_pred = [stop == "Max Epochs" for stop in STOP_pred].sum() / brain_areas
    STOP_clas = [stop == "Max Epochs" for stop in STOP_clas].sum() / brain_areas
    return [mean_train_pred, std_train_pred, mean_train_clas, std_train_clas, STOP_pred, STOP_clas, area_accuracy, true_positive, true_negative, mean_mse, std_mse]