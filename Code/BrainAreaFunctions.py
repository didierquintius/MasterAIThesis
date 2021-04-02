# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:40:23 2021

@author: Quintius
"""

from DataSimulation_functions import simulateData
import numpy as np
from Visualize_functions import plot_line
from NeuralNetworks import NN, CNN
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#%%

def train_brain_area(brain_area, param_values, val_perc = 0.1, plot = False, add_networks = False):
     
    def train_PredictionModel(param_values, EEG_Data, source_activity):
        
        electrodes = EEG_Data.shape[1]
        
        # shuffle data
        shuffled_indexes = np.random.permutation(EEG_Data.shape[0])
        EEG_Data = EEG_Data[shuffled_indexes, :]
        source_activity = source_activity[shuffled_indexes, :]
        
        
        #split data into validation and training sets
        X_train,X_val,y_train,y_val = train_test_split(EEG_Data,source_activity,test_size = val_perc)
        
        # initialize Neuralnetwork and relevant parameters
        Net = NN(electrodes, param_values['nodes_pred'], 1)
        
        Net.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        history = Net.fit(X_train,y_train, epochs=param_values['EPOCHS_pred'], batch_size=param_values['batch_sizes_pred'],
                validation_data=(X_val, y_val), callbacks = [es], verbose = 0)
        
        return Net, history
        
    def trainClassificationModel(param_values, activity, class_trial):
       
        #split data into validation and training sets
        X_train,X_val,y_train,y_val = train_test_split(activity, class_trial,test_size = val_perc)

        Net = CNN(param_values['time_steps'], param_values['nodes_Conv_clas'], param_values['nodes_Dense_clas'],
                  param_values['kernel_size'], param_values['strides'], 1)
        Net.compile(optimizer = 'adam',  loss='binary_crossentropy',  metrics=['accuracy','mse'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        history = Net.fit(X_train,y_train, epochs=param_values['EPOCHS_clas'], batch_size=param_values['batch_sizes_clas'],
                validation_data=(X_val, y_val), callbacks = [es], verbose = 0)
        
        return Net, history 
            
    
    # simulate data
    EEG_Data, activity, source_trials, noisy_trials = simulateData(param_values,'train',train_dipole= brain_area)
    electrodes, time_steps, _ = EEG_Data.shape
    

    # format  prediction data
    EEG_Data = EEG_Data.transpose().reshape(-1, electrodes)
    activity = activity.reshape(-1, 1)
    source_trials = source_trials.reshape(-1, 1)
    NN_Net, history_pred = train_PredictionModel(param_values, EEG_Data, activity)
    
    # format classification data
    predicted_source_activity = NN_Net.predict(EEG_Data).reshape((-1, time_steps, 1))
    
    CNN_Net, history_clas = trainClassificationModel(param_values, predicted_source_activity, source_trials)
    
    
    EEG_Data, activity, source_trials, noisy_trials  = simulateData(param_values,'train', train_dipole = brain_area, seed = 1)
    
    
    predicted_source_activity = NN_Net.predict(EEG_Data.transpose().reshape(-1, electrodes))
    predicted_class = CNN_Net.predict(predicted_source_activity.reshape(-1, time_steps, 1))
    
    
    mse_pred = np.mean(history_pred.history['val_mse'])
    mse_clas = np.mean(history_clas.history['val_mse'])

    true_active_clas = (predicted_class[source_trials == 1] >= 0.5).mean()
    true_noisy_clas = (predicted_class[noisy_trials == 1] < 0.5).mean()
    true_idle_clas = (predicted_class[(source_trials == 0) & (noisy_trials == 0)] < 0.5).mean()
    accuracy = (true_active_clas + true_noisy_clas + true_idle_clas) / 3
    if plot:
        activity = activity.reshape((-1))
        predicted_source_activity = predicted_source_activity.reshape((-1))
        plot_line([activity, predicted_source_activity],['Original','Predicted'], title = 'Prediction_Performance')
        source_trials = np.repeat(source_trials, time_steps).reshape((-1)) * 1
        predicted_class = np.repeat(predicted_class, time_steps).reshape((-1))
        plot_line([predicted_source_activity, source_trials, predicted_class],['Predicted Activity', 'Original State', 'Predicted State'], dash = ['solid', 'dash', 'dot'],title = 'Classification Performance')
        plot_line([np.array(history_pred.history['val_mse']), np.array(history_pred.history['mse'])],
                            ['Prediction Validation Loss','Prediction Training Loss'],
                            title = 'Prediction Loss')
        plot_line([np.array(history_clas.history['val_mse']), np.array(history_clas.history['mse'])],
                    ['Classification Validation Loss','Classification Training Loss'],
                    title = 'Classification Loss')    
    if add_networks:
        data = (mse_pred, mse_clas, true_active_clas, true_noisy_clas, true_idle_clas, accuracy, NN_Net, CNN_Net)
    else: 
        data = (mse_pred, mse_clas, true_active_clas, true_noisy_clas, true_idle_clas, accuracy)
    return data
    
    
    
    
    
