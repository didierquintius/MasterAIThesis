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

def importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials, seed = 0):
        data_file = "../Data/EEG/data_" + str(snr) + "_" + str(cnr) + "_" + str(noisy_areas) + "_" + str(brain_areas) + "_" + str(time_steps) + "_" + str(trials) + ".pkl"
        
        if os.path.exists(data_file):            
            EEG_data, sources, activity = pickle.load(open(data_file, "rb"))
        else:
            EEG_data, sources, noisy_sources, activity =  EEG_signal(time_steps, trials, brain_areas,
                                                      snr, cnr, noisy_areas, seed)
        
        return EEG_data, sources, noisy_sources, activity


def runModel(snr, cnr, noisy_areas, brain_areas, params, plot = False, prev_data_pred = None, prev_data_clas = None, seed = 0):
    
    time_steps, trials, pred_arch, clas_arch, kernel, stride, preportion_pred, preportion_clas, lr_pred, lr_clas, batch_pred, batch_clas, val_treshold_pred, val_treshold_clas = params
    #%%
    
    EEG_data, sources, noisy_sources, activity =  importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials, seed)
    EEG_data_test,  sources_test, noisy_sources_test, activity_test, EEG_data_trainval,  sources_trainval, noisy_sources_trainval, activity_trainval = splitTestData(EEG_data, sources, noisy_sources, activity)
    del EEG_data, sources, activity
    
    #%%
    NeuralNets, train_pred, val_losses_pred, STOP_pred = fitProjectionModels(EEG_data_trainval, sources_trainval, noisy_sources_trainval, activity_trainval,
                                                                                                  brain_areas, architecture = pred_arch, preportion = preportion_pred, 
                                                                                                  batch_size = batch_pred, learning_rate = lr_pred, val_freq = val_treshold_pred, prev_nets = prev_data_pred)
    
    #%%
    CNN_Nets, train_clas, val_losses_clas, STOP_clas = fitClassificationModels(EEG_data_trainval,sources_trainval,brain_areas,NeuralNets, clas_arch, kernel, stride,
                                                                                                    lr_clas, preportion_clas, batch_size = batch_clas, val_freq = val_treshold_clas, prev_nets = prev_data_clas)
    area_accuracy, true_positive, true_negative, mean_mse, std_mse, _ = test_performance(EEG_data_test, sources_test, activity_test, brain_areas, NeuralNets, CNN_Nets, plot)
    
    STOP_pred_mean = 1 - sum([stop == "Max Epochs" for stop in STOP_pred]) / brain_areas
    STOP_clas_mean = 1 - sum([stop == "Max Epochs" for stop in STOP_clas]) / brain_areas
    mean_train_pred = train_pred.mean()
    std_train_pred = train_pred.std()
    mean_train_clas = train_clas.mean()
    std_train_clas = train_clas.std()    
    return [time_steps, trials, pred_arch[0], clas_arch[0], clas_arch[1], kernel[0], stride[0], lr_pred, lr_clas,
            preportion_pred, preportion_clas, batch_pred, batch_clas, val_treshold_pred,
            val_treshold_clas, mean_train_pred, std_train_pred, mean_train_clas, std_train_clas, 
            STOP_pred_mean, STOP_clas_mean, area_accuracy, true_positive, true_negative, mean_mse, std_mse], NeuralNets, STOP_pred, CNN_Nets, STOP_clas, train_pred, train_clas