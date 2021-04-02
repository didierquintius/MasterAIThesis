# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:36:54 2020

@author: didie
"""

import pickle, random, os
import numpy as np
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%

def setNNFormat(data, nn_input):
    data = torch.Tensor(data.reshape((nn_input, -1),order = "F").transpose())
    return data
def generate(seed, time_steps, trials, no_active_sources = 3):
    # generates activitactivity for n sources for a given number of timesteps and trials
    
    np.random.seed(seed)
    activity = np.zeros((no_active_sources, time_steps,trials))
    # set every first time step to zero and every second time step to 1
    activity[:,1,:] = 1
    parameters = np.zeros((3,3,2))
    # set parameters for 
    parameters[0,:,:] = [[0.5, -0.7],[0, 0],[0, 0]]
    parameters[1,:,:] = [[0.2, 0],[0.7, -0.5],[0, 0]]
    parameters[2,:,:] = [[0, 0],[0, 0],[0, 0.8]]
    for trial in range(trials):
        for time_step in range(2, time_steps):
            if time_step < time_steps / 2:
                parameters[0,1,0] = 0.5 * (time_step/ time_steps)
            else:
                parameters[0,1,0] = 0.5* (time_steps - time_step) / (time_steps / 2)
            
            parameters[1,2,0] = 0.4 if time_step < 0.7 * time_steps else 0
            
            # multiply the previous values of the brain activity with the parameters
            # add random noise to obtain new values of the brain acitivy
            activity[:, time_step, trial] = np.einsum('ijk,jk', parameters, activity[:,[time_step-1,time_step-2],trial]) + np.random.random(3) - 0.5
   
    return activity         

#%%
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

def EEG_signal(time_steps,trials,no_brain_areas, sig_noise_ratio , channel_noise_ratio,
                noise_sources, seed = 0, only_save = False):
    
    # load projection matrix and subset to the relevant brain areas
    projection_matrix = pickle.load(open( os.environ['DATA'] + "/MasterAIThesis/projection_matrix.pkl", "rb" ))
    projection_matrix = projection_matrix[:, range(0,projection_matrix.shape[1],
                                                   int(projection_matrix.shape[1]/(no_brain_areas - 1)))]

    # determine which brain areas will be active during each trial
    sample_list = np.arange(no_brain_areas).tolist()
    active_brain_areas = []
    noisy_brain_areas = []
    # each brain area is active part 1 equal amount of times
    # the other two active no_brain_areas are chosen randomly
    for neuron in range(no_brain_areas):
        temp_sample_list = sample_list.copy()
        temp_sample_list.remove(neuron)
        amount = int(trials/no_brain_areas * (neuron + 1)) - int(trials/no_brain_areas * neuron)
        active_brain_areas = active_brain_areas + [[neuron] + random.sample(temp_sample_list, 2) for i in range(amount)]
    no_electrodes = projection_matrix.shape[0]
    EEG_Data = np.zeros((no_electrodes, time_steps, trials))
    activity = generate(seed, time_steps, trials)
    
    for trial in range(trials):
        
        noisy_activity = standardize(pink_noise(time_steps, noise_sources))
        source_activity_trial = standardize(activity[:,:,trial])
        noisy_elements = np.random.choice(no_brain_areas, noise_sources, replace = False)
        noisy_brain_areas +=  [noisy_elements]
        
        EEG_activity = projection_matrix[:, active_brain_areas[trial]] @ source_activity_trial
        EEG_noisy = projection_matrix[:, noisy_elements] @ noisy_activity
        EEG_signal = standardize(sig_noise_ratio * EEG_activity + (1 - sig_noise_ratio) * EEG_noisy)
        EEG_channel_noise = np.random.normal(size = (no_electrodes, time_steps))
        
        EEG_Data[:,:,trial] = channel_noise_ratio * EEG_signal + (1 - channel_noise_ratio) * EEG_channel_noise
        activity[:,:,trial] = source_activity_trial
    
    active_brain_areas = np.array(active_brain_areas)
    noisy_brain_areas = np.array(noisy_brain_areas)
    shuffled_indexes = np.arange(trials).tolist()
    random.shuffle(shuffled_indexes)
    EEG_Data = EEG_Data[:,:,shuffled_indexes]
    activity = activity[:,:,shuffled_indexes]
    active_brain_areas = active_brain_areas[shuffled_indexes, :]
    noisy_brain_areas = noisy_brain_areas[shuffled_indexes, :]
    data = (EEG_Data, active_brain_areas, noisy_brain_areas, activity)
    pickle.dump(data, open(os.environ['DATA'] + "/MasterAIThesis/EEG/data_" + str(sig_noise_ratio) + "_" + str(channel_noise_ratio) + "_" + str(noise_sources) + "_" + str(no_brain_areas) + "_" + str(time_steps) + "_" + str(trials)+ ".pkl", "wb" ))
    
    if not only_save:
        return data
    


def Balanced_EEG(params, relevant_brain_area, sig_noise_ratio = 0.9, channel_noise_ratio = 0.9, seed = 0, only_save = False):
    time_steps,trials,no_brain_areas, noise_sources = params['time_steps'], params['trials'], params['brain_areas'], int(0.5 * params['brain_areas'])
    file_name = os.environ['DATA'] + "/MasterAIThesis/Training/data_" + str(sig_noise_ratio) + "_" + str(channel_noise_ratio) + "_" + str(no_brain_areas) + "_" + str(time_steps) + "_" + str(trials) + "_" + str(relevant_brain_area) + "_" + str(seed) + ".pkl"
    if os.path.isfile(file_name):
        return pickle.load(open(file_name, "rb"))
    
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
        
    
    def active_areas(trial):
        if trial < int(trials / 9):
            return [relevant_brain_area] + random.sample(non_relevant_brain_areas, 2)
        elif trial < int(trials * 2 / 9):
            samples = random.sample(non_relevant_brain_areas, 2)
            return [samples[0]] + [relevant_brain_area] + [samples[1]]
        elif trial < int(trials / 3):
            return random.sample(non_relevant_brain_areas, 2) + [relevant_brain_area] 
        else:
            return random.sample(non_relevant_brain_areas, 3)
    
    # load projection matrix and subset to the relevant brain areas
    projection_matrix = pickle.load(open( os.environ['DATA'] + "/MasterAIThesis/projection_matrix.pkl", "rb" ))
    projection_matrix = projection_matrix[:, range(0,projection_matrix.shape[1],
                                                   int(projection_matrix.shape[1]/(no_brain_areas - 1) - 1))]
    

    active_brain_areas = []
    noisy_brain_areas = []
    non_relevant_brain_areas = np.delete(np.arange(no_brain_areas), relevant_brain_area).tolist()
    
    no_electrodes, _ = projection_matrix.shape
    EEG_Data = np.zeros((no_electrodes, time_steps, trials))
    source_activity = generate(seed, time_steps, trials)
    noisy_activity = np.zeros((noise_sources,time_steps, trials))

    for trial in range(trials):
        active_sources =  active_areas(trial)
        active_brain_areas += [active_sources]

        non_actives = np.delete(np.arange(no_brain_areas), active_sources).tolist()
        noisy_sources = random.sample(non_actives, noise_sources)  
        noisy_brain_areas += [noisy_sources]
        
        noisy_activity_trial = standardize(pink_noise(time_steps, noise_sources))
        source_activity_trial = standardize(source_activity[:,:,trial])
        
        EEG_activity = projection_matrix[:, active_sources] @ source_activity_trial
        EEG_noisy =    projection_matrix[:, noisy_sources]  @ noisy_activity_trial
        
        EEG_channel_noise = np.random.normal(size = (no_electrodes, time_steps))
        
        EEG_signal = standardize(sig_noise_ratio * EEG_activity + (1 - sig_noise_ratio) * EEG_noisy)

        EEG_Data[:,:,trial] = channel_noise_ratio * EEG_signal + (1 - channel_noise_ratio) * EEG_channel_noise
        source_activity[:,:,trial] = source_activity_trial
        noisy_activity[:,:,trial]= noisy_activity_trial
        
    
    active_brain_areas = np.array(active_brain_areas)
    noisy_brain_areas = np.array(noisy_brain_areas)
    shuffled_indexes = np.arange(trials).tolist()
    random.shuffle(shuffled_indexes)
    eeg_max = max([-EEG_Data.min(), EEG_Data.max()])
    EEG_Data = EEG_Data[:,:,shuffled_indexes] / eeg_max
    source_activity = source_activity[:,:,shuffled_indexes]
    noisy_activity =  noisy_activity[:,:,shuffled_indexes]
    active_brain_areas = active_brain_areas[shuffled_indexes, :]
    noisy_brain_areas = noisy_brain_areas[shuffled_indexes, :]
    activity, source_trials, noisy_trials = filterCorrespondingData(source_activity, active_brain_areas, noisy_activity, noisy_brain_areas, relevant_brain_area)
    activity_max = max([activity.max(), -activity.min()])
    activity = activity / activity_max
    data = (EEG_Data, activity, source_trials, noisy_trials)
    pickle.dump(data, open(file_name, "wb" ))
    return data

    

