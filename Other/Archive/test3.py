# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:54:13 2021

@author: didie
"""

import pickle, random, os
import numpy as np
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
def setNNFormat(data, nn_input):
    # reshape data to the correct input shape for the neural network
    data = torch.Tensor(data.reshape((nn_input, -1),order = "F").transpose())
    return data

def generate(seed, time_steps, trials, n_source_sources = 3):
    # generates neural activity for n sources for a given number of timesteps and trials

    np.random.seed(seed)
    activity = np.zeros((n_source_sources, time_steps,trials))
    # set every first time step to zero and every second time step to 1
    activity[:,1,:] = 1
    parameters = np.zeros((3,3,2))
    # set parameters for the AR process
    parameters[0,:,:] = [[0.5, -0.7],[0, 0],[0, 0]]
    parameters[1,:,:] = [[0.2, 0],[0.7, -0.5],[0, 0]]
    parameters[2,:,:] = [[0, 0],[0, 0],[0, 0.8]]
    
    for trial in range(trials):
        for time_step in range(2, time_steps):
            # the size of two parameters are depended on the timesteps and are
            # calculated using the if statements
            if time_step < time_steps / 2:
                parameters[0,1,0] = 0.5 * (time_step / time_steps)
            else:
                parameters[0,1,0] = 0.5* (time_steps - time_step) / (time_steps / 2)
            
            parameters[1,2,0] = 0.4 if time_step < 0.7 * time_steps else 0
            
            # multiply the previous values of the brain activity with the parameters
            # add random noise (between -0.5 and 0.5) to obtain new values of the brain acitivy
            activity[:, time_step, trial] = np.einsum('ijk,jk', parameters, activity[:,[time_step-1,time_step-2],trial]) + np.random.random(3) - 0.5
   
    return activity       

def pink_noise(n, m):
    n1 = n + 1 if n % 2 == 0 else n
    mid_point = int((n1 - 1) / 2)
    scal = np.sqrt(1/np.arange(1,mid_point))
    ff=np.zeros((m, n1))

    ff[:, 1:mid_point]=np.resize(scal,(m, mid_point -1))
    
    noise=np.fft.fft(np.random.normal(size = (n1,m)))
    combi = np.transpose(np.multiply(noise, np.transpose(ff)))
    noise=2*np.real(np.fft.ifft(combi))
    noise=noise[:, :n]
    return noise

def standardize(x):
    return (x - x.mean())/x.std()

def filterCorrespondingData(source_activity, source_dipoles, noisy_activity, noisy_dipoles, rel_dipole):
        def setNNFormat(data, nn_input):
            data = torch.Tensor(data.reshape((nn_input, -1),order = "F").transpose())
            return data
    
        def keep_relevant(activity, dipoles, rel_dipole):
            state_trials = np.array([[dipole == rel_dipole for dipole in sources] for sources in dipoles]).T
            activity = activity.swapaxes(1,0)
            activity = np.sum(activity * state_trials, axis = 1)
            state_trials = state_trials.sum(axis = 0)
            return activity, state_trials
        
        source_activity, source_trials = keep_relevant(source_activity, source_dipoles, rel_dipole)
        noisy_activity, noisy_trials = keep_relevant(noisy_activity, noisy_dipoles, rel_dipole)
        activity = source_activity + noisy_activity
    
        return activity, source_trials, noisy_trials
    
def source_areas(trial, trials, non_relevant_dipoles, relevant_dipole):
    if trial < int(trials / 9):
        return [relevant_dipole] + random.sample(non_relevant_dipoles, 2)
    elif trial < int(trials * 2 / 9):
        samples = random.sample(non_relevant_dipoles, 2)
        return [samples[0]] + [relevant_dipole] + [samples[1]]
    elif trial < int(trials / 3):
        return random.sample(non_relevant_dipoles, 2) + [relevant_dipole] 
    else:
        return random.sample(non_relevant_dipoles, 3)

def train_active_dipoles(trials, n_relevant_dipoles, train_dipole):
    # select 3 dipoles as active during each trials, ensure equal presentation
    # by generating a random order of the dipoles and reshaping this in
    # triplets, when the trials are more than the number of relevant dipoles
    # repeat the process
    
    non_train_dipoles = np.delete(np.arange(n_relevant_dipoles), train_dipole)
    active_trials = int(trials / 3)
    other_trials = trials - active_trials
    n_non_train = len(non_train_dipoles)
    
    # the amount of dipole states that need to be determined
    dipole_selections_active = active_trials * 2
    dipole_selections_other = other_trials * 3
    
    # calculate amount of dipoles that can simulated simultaneously
    max_permutation_size_active = n_non_train - n_non_train % 2
    max_permutation_size_other  = n_non_train - n_non_train % 3
    
    # split trials up into batches of the maximum size of each iteration
    permutation_sizes_active = [max_permutation_size_active] * int(dipole_selections_active / max_permutation_size_active)
    permutation_sizes_active += [dipole_selections_active% max_permutation_size_active]
    
    permutation_sizes_other = [max_permutation_size_other] * int(dipole_selections_other / max_permutation_size_other)
    permutation_sizes_other += [dipole_selections_other % max_permutation_size_other]
    
    # select which dipoles are the source during each trial
    source_dipoles = []
    for size in permutation_sizes_other:
        new_source_dipoles = np.random.permutation(non_train_dipoles[:size])
        source_dipoles += new_source_dipoles.reshape((-1, 3)).tolist()
        
    active_source_dipoles = np.ones((active_trials, 3)).astype('int') * -1
    active_source_dipoles[:int(active_trials/3),0] = train_dipole
    active_source_dipoles[int(active_trials/3):int(active_trials*2/3),1] = train_dipole
    active_source_dipoles[int(active_trials*2/3):, 2] = train_dipole
    empty_coord = np.where(active_source_dipoles == -1)
    
    for size in permutation_sizes_active:
        new_source_dipoles = np.random.permutation(non_train_dipoles[:size])
        empty_coord = np.where(active_source_dipoles == -1)
        relevant_coord = (empty_coord[0][:size], empty_coord[1][:size])
        active_source_dipoles[relevant_coord] = new_source_dipoles
        
    source_dipoles += active_source_dipoles.tolist()
    shuffled_indexes = np.random.permutation(trials)
    source_dipoles = source_dipoles[shuffled_indexes]
    
    return source_dipoles

def standard_active_dipoles(trials, n_relevant_dipoles):
    # select 3 dipoles as active during each trials, ensure equal presentation
    # by generating a random order of the dipoles and reshaping this in
    # triplets, when the trials are more than the number of relevant dipoles
    # repeat the process
    
    # the amount of dipole states that need to be determined
    dipole_selections = trials * 3
    
    # calculate amount of dipoles that can simulated simultaneously
    max_permutation_size = n_relevant_dipoles - (n_relevant_dipoles % 3)
    
    # split trials up into batches of the maximum size of each iteration
    permutation_sizes = [max_permutation_size] * int(dipole_selections / max_permutation_size)
    
    # add leftover dipole selections
    permutation_sizes += [(dipole_selections) % max_permutation_size]
    
    # select which dipoles are the source during each trial
    source_dipoles = []
    for size in permutation_sizes:
        new_source_dipoles = np.random.permutation(size)
        source_dipoles += new_source_dipoles.reshape((-1, 3)).tolist()
    return source_dipoles

def close_active_dipoles(trials, n_relevant_dipoles = 1000):
    
    distances = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\dipole_distances.pkl', 'rb'))
    close_dipoles = []
    for dipole in range(1000):
        close_dipoles += [np.where(distances[dipole,:] < 20)[0].tolist()]
    source_dipoles = []
    for trial in range(trials):
        random.seed(int(trial / n_relevant_dipoles))
        dipole = trial % n_relevant_dipoles
        source_dipoles += [[dipole] + random.sample(close_dipoles[dipole], 2)]
    return source_dipoles
def simulateTrialData(source_dipoles, source_activity, params , projection_matrix,
                      sig_noise_ratio, channel_noise_ratio):

    time_steps, n_relevant_dipoles, n_noise_dipoles = params['time_steps'], params['brain_areas'], int(0.5 * params['brain_areas'])
    n_electrodes = projection_matrix.shape[0]
    # sample the noisy dipoles from the remaining not source dipoles
    non_source_dipoles = np.delete(np.arange(n_relevant_dipoles), source_dipoles).tolist()
    noisy_dipoles = random.sample(non_source_dipoles, n_noise_dipoles)  
          
    # generate noisy activity
    noisy_activity = pink_noise(time_steps, n_noise_dipoles)
          
    # project the neural activity to the scalp
    EEG_source = projection_matrix[:, source_dipoles] @ source_activity
    EEG_noisy =  projection_matrix[:, noisy_dipoles]  @ noisy_activity
    
    # standerdize neural activity before and after sum to ensure accurate ratio
    EEG_source = standardize(EEG_source)
    EEG_noisy =  standardize(EEG_noisy)
    EEG_signal_pure = standardize(sig_noise_ratio * EEG_source + (1 - sig_noise_ratio) * EEG_noisy)
    
    # add channel noise over the pure EEG signal
    EEG_channel_noise = np.random.normal(size = (n_electrodes, time_steps))
    EEG_signal = channel_noise_ratio * EEG_signal_pure + (1 - channel_noise_ratio) * EEG_channel_noise
    
    return EEG_signal, noisy_activity, noisy_dipoles  
                 
def simulateData(params, simulation, train_dipole = None, sig_noise_ratio = 0.9, channel_noise_ratio = 0.9, seed = 0, only_save = False):
    # get parameters
    time_steps,trials,n_relevant_dipoles = params['time_steps'], params['trials'], params['brain_areas']
    
    # check if there already exist a dataset with the same specifications
    file_name = os.environ['DATA'] + "/MasterAIThesis/Training/data_" + str(sig_noise_ratio) + "_" + str(channel_noise_ratio) + "_" + str(n_relevant_dipoles) + "_" + str(time_steps) + "_" + str(trials) + "_" + str(simulation) + "_" + str(seed) + ".pkl"
    if os.path.isfile(file_name):
        return pickle.load(open(file_name, "rb"))
    
    # load projection matrix and subset to the relevant brain areas
    projection_matrix = pickle.load(open( os.environ['DATA'] + "/MasterAIThesis/projection_matrix.pkl", "rb" ))
    n_electrodes, total_dipoles = projection_matrix.shape
    relevant_dipoles = range(0,total_dipoles, int(total_dipoles/(n_relevant_dipoles - 1) - 1))[:n_relevant_dipoles]
    projection_matrix = projection_matrix[:, relevant_dipoles]
    
    if simulation == 'standard': 
        source_dipoles = standard_active_dipoles(trials, n_relevant_dipoles)
    elif simulation == 'train':
        source_dipoles = train_active_dipoles(trials, n_relevant_dipoles, train_dipole)
    elif simulation == 'close':
        source_dipoles = close_active_dipoles(trials)
    else:
        print('simulation strategy not available')
        return

    EEG_Data = np.zeros((n_electrodes, time_steps, trials))
    source_activity = generate(seed, time_steps, trials)
    noisy_dipoles = []
    activity = np.zeros((n_relevant_dipoles, trials, time_steps))
    
    for trial in range(trials):
        EEG_signal, noisy_activity_trial, noisy_dipoles_trial = simulateTrialData(source_dipoles[trial], source_activity[:, :, trial], params, 
                                                                                  projection_matrix, sig_noise_ratio, channel_noise_ratio)
        # add trial data to total data set
        EEG_Data[:,:,trial] = EEG_signal
        noisy_dipoles += [noisy_dipoles_trial]    
        
        activity[source_dipoles[trial], trial, :] = source_activity[:, :, trial]
        activity[noisy_dipoles_trial,   trial, :] = noisy_activity_trial
              
    if simulation == 'train':
        activity = activity[train_dipole, :, :]
        
    source_dipoles = np.array(source_dipoles)
    noisy_dipoles = np.array(noisy_dipoles)
    
    # fit all the EEG and activity data between -1 and 1
    eeg_max = max([-EEG_Data.min(), EEG_Data.max()])
    EEG_Data = EEG_Data / eeg_max

    activity_max = max([activity.max(), -activity.min()])
    activity = activity / activity_max
    
    data = (EEG_Data, activity, source_dipoles, noisy_dipoles)
    pickle.dump(data, open(file_name, "wb" ))
    return data