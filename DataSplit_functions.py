# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:28 2020

@author: didie
"""
import pickle, torch, os, random
import numpy as np
from itertools import compress
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
def splitData(EEG_data, sources, activity, brain_areas, train_perc = 0.7,
                      val_perc = 0.1):  
    
    elecs, time_steps, trials = EEG_data.shape
        
    XandY  = {"y": {}, "X": {}}
    original_data = {"X": EEG_data, "y" : activity}
    no_cols = {"X" : elecs, "y" : 1}
    
    for XorY, data in XandY.items():
        for brain_area in range(brain_areas):
            data[str(brain_area)]= {}
            relevant_trials = [brain_area in source for source in sources]
            relevant_data = original_data[XorY][:, :, relevant_trials]
            total_trials = range(sum(relevant_trials))
            random.seed(0)
            train = random.sample(total_trials, int(train_perc * len(total_trials)),)
            remaining_trials = list(set(total_trials) -set(train))
            val = random.sample(remaining_trials, int(val_perc * len(total_trials)),)
            test = list(set(total_trials) - set(val))

            indexes = {"train": train, "val":  val,  "test": test}
          
            if XorY == "y":
                relevant_sources = list(compress(sources, relevant_trials))
                relevant = [[neuron == brain_area for neuron in source] for source in relevant_sources]
                relevant_data = relevant_data.swapaxes(1,0)
                relevant_data = np.sum(relevant_data * np.array(relevant).T, axis = 1)
            
                for partition_name, index in indexes.items():
                    data[str(brain_area)][partition_name] = torch.Tensor(relevant_data[:, index].copy().reshape(no_cols[XorY], time_steps * len(index), order = "F"))
                    data[str(brain_area)][partition_name] = data[str(brain_area)][partition_name].transpose(0,1)

            else:
                for partition_name, index in indexes.items():
                    data[str(brain_area)][partition_name] = torch.Tensor(relevant_data[:,:,index].copy().reshape(no_cols[XorY], time_steps * len(index), order = "F"))
                    data[str(brain_area)][partition_name] = data[str(brain_area)][partition_name].transpose(0,1)
      
    return XandY["X"], XandY["y"]

def splitTestData(EEG_data, sources, activity, test_perc = 0.2):
    electrodes, timesteps, trials = EEG_data.shape
    
    # calculate number of trials in the test data
    test_ind = int(trials * test_perc)
    
    EEG_data_test = EEG_data[:,:,:test_ind]
    activity_test = activity[:,:,:test_ind]
    sources_test = sources[:test_ind,:]
    
    EEG_data_trainval = EEG_data[:,:,test_ind:]
    activity_trainval = activity[:,:,test_ind:]
    sources_trainval = sources[test_ind:,:]
    
    return EEG_data_test,  sources_test,activity_test, EEG_data_trainval,  sources_trainval, activity_trainval

def setNNFormat(data, nn_input):
        data = torch.Tensor(data.reshape((nn_input, -1),order = "F").transpose())
        return data
    
def prepareProjectionData(EEG_data, sources, activity, brain_area, train_perc = 0.7, val_perc = 0.1):

    
    def filterActivityData(data, relevant_trials, sources = sources, brain_area = brain_area):
        data = data[:, :, relevant_trials]
        relevant_sources = sources[relevant_trials].tolist()
        relevant_data_indexes = [[neuron == brain_area for neuron in source] for source in relevant_sources]
        data = data.swapaxes(1,0)
        data = np.sum(data * np.array(relevant_data_indexes).T, axis = 1)
        data = setNNFormat(data, 1)
        return data
    
    
    
    elecs = EEG_data.shape[0]
    # find the trials where the brain area was active
    relevant_trials = np.where([brain_area in source for source in sources.tolist()])[0]
    
    # calculate the amount of trails in the train data
    train_ind = int(train_perc /(train_perc + val_perc ) * len(relevant_trials))
    
    EEG_train = setNNFormat(EEG_data[:, :, relevant_trials[:train_ind]], elecs)
    EEG_val = setNNFormat(EEG_data[:, :, relevant_trials[train_ind:]], elecs)
    
    activity_train  = filterActivityData(activity, relevant_trials[:train_ind])  
    activity_val  = filterActivityData(activity, relevant_trials[train_ind:])
        
    return EEG_train, activity_train, EEG_val, activity_val
    
def prepareClassificationData(EEG_data, sources, brain_areas, NeuralNets, train_perc = 0.7, val_perc = 0.1):
    random.seed(0)
    electrodes, time_steps, trials = EEG_data.shape
    NNinput_EEG = setNNFormat(EEG_data, electrodes)

    
    brain_area_activity = torch.Tensor(np.zeros((trials * brain_areas, 1)))
    for trial, active_areas in enumerate(sources.tolist()):
        for area in active_areas:
            brain_area_activity[area * trials + trial ,0] = 1
    
    activity_prediction = torch.Tensor()
    for brain_area in range(brain_areas):
        prediction = NeuralNets[brain_area](NNinput_EEG)
        activity_prediction = torch.cat((activity_prediction, prediction))
        
    activity_prediction = activity_prediction.reshape((-1, 1, time_steps)) 
    
    active_time_series = np.where(brain_area_activity == 1)[0]
    idle_time_series =  np.where(brain_area_activity == 0)[0]
    
    val_ind = int(val_perc /(train_perc + val_perc ) * len(active_time_series))
    
    val_indexes = np.append(active_time_series[:val_ind],idle_time_series[:val_ind])
    
    X = {}
    y = {}
    
    X["val"]= activity_prediction[val_indexes,:]
    X["train"] = {}
    X["train"]["active"] = activity_prediction[active_time_series[val_ind:],:,:]
    X["train"]["idle"] = activity_prediction[idle_time_series[val_ind:],:,:]
    
    y["val"] = brain_area_activity[val_indexes,:]
    y["train"] = {}
    y["train"]["active"] = brain_area_activity[active_time_series[val_ind:],:]
    y["train"]["idle"] = brain_area_activity[idle_time_series[val_ind:],:]

    return X, y