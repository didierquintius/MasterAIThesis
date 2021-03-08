# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:14 2020
@author: didie
"""
import pickle, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from Fit_functions import fitProjectionModels, fitClassificationModels
from DataSimulation_functions import EEG_signal
from DataSplit_functions import splitTestData
from performance_function import test_performance
#%%
def importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials, seed = 0):
    
    EEG_data, sources, noisy_sources, activity =  EEG_signal(time_steps, trials, brain_areas,
                                                      snr, cnr, noisy_areas, seed)
        
    return EEG_data, sources, noisy_sources, activity


def runModel(snr, cnr, noisy_areas, brain_areas, params, plot = False, seed = 0, test_trials = 1000):
    
    time_steps, trials, pred_arch, clas_arch, kernel, stride, preportion_pred, preportion_clas, lr_pred, lr_clas, batch_pred, batch_clas, val_treshold_pred, val_treshold_clas, brain_areas, max_val_amount, val_freq_pred, val_freq_clas = params
    #%%
    EEG_data_trainval,  sources_trainval, noisy_sources_trainval, activity_trainval = importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials, seed + 1)


    
    #%%
    NeuralNets, train_pred, val_losses_pred, STOP_pred = fitProjectionModels(EEG_data_trainval, sources_trainval, noisy_sources_trainval, activity_trainval,
                                                                                                  brain_areas, architecture = pred_arch, preportion = preportion_pred, 
                                                                                                  batch_size = batch_pred, learning_rate = lr_pred, min_increment = val_treshold_pred, val_freq = val_freq_pred, max_val_amount = max_val_amount)
    
    #%%
    CNN_Nets, train_clas, val_losses_clas, STOP_clas = fitClassificationModels(EEG_data_trainval,sources_trainval,brain_areas,NeuralNets, clas_arch, kernel, stride,
                                                                                                    lr_clas, preportion_clas, batch_size = batch_clas, min_increment = val_treshold_clas, val_freq=val_freq_clas, max_val_amount = max_val_amount)
    
    del EEG_data_trainval,  sources_trainval, noisy_sources_trainval, activity_trainval 
    EEG_data_test,  sources_test, noisy_sources_test, activity_test,  = importData(snr, cnr, noisy_areas, brain_areas, time_steps, test_trials, 0)
    area_accuracy, true_positive, true_negative, mean_mse, std_mse = test_performance(EEG_data_test, sources_test, activity_test, brain_areas, NeuralNets, CNN_Nets, plot)
    
    STOP_pred_mean = 1 - sum([stop == "Max Epochs" for stop in STOP_pred]) / brain_areas
    STOP_clas_mean = 1 - sum([stop == "Max Epochs" for stop in STOP_clas]) / brain_areas
    mean_train_pred = train_pred.mean()
    std_train_pred = train_pred.std()
    mean_train_clas = train_clas.mean()
    std_train_clas = train_clas.std()    
    return [time_steps, trials, pred_arch[0], clas_arch[0], clas_arch[1], kernel[0], stride[0], lr_pred, lr_clas,
            preportion_pred, preportion_clas, batch_pred, batch_clas, val_treshold_pred,
            val_treshold_clas, brain_areas, max_val_amount, val_freq_pred, val_freq_clas, mean_train_pred, std_train_pred, mean_train_clas, std_train_clas, 
            STOP_pred_mean, STOP_clas_mean, area_accuracy, true_positive, true_negative, mean_mse, std_mse], (NeuralNets, CNN_Nets), (val_losses_pred, val_losses_clas)