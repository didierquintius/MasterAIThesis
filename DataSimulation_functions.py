# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:36:54 2020

@author: didie
"""

import pickle, random, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
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
    
    noise=np.fft.fft(np.random.normal(size = (n1,m)),axis = 0)
    noise=2*np.real(np.fft.ifft(np.transpose(np.multiply(noise, np.transpose(ff))), axis = 1))
    noise=noise[:, :n]
    return noise

def EEG_signal(time_steps,trials,no_brain_areas, sig_noise_ratio , channel_noise_ratio,
                noise_sources, seed = 0, only_save = False):
    
    # load projection matrix and subset to the relevant brain areas
    projection_matrix = pickle.load(open( "../../Data/projection_matrix.pkl", "rb" ))
    projection_matrix = projection_matrix[:, range(0,projection_matrix.shape[1],
                                                   int(np.ceil(projection_matrix.shape[1]/no_brain_areas)))]

    # determine which brain areas will be active during each trial
    sample_list = np.arange(no_brain_areas).tolist()
    active_brain_areas = []
    # each brain area is active part 1 equal amount of times
    # the other two active no_brain_areas are chosen randomly
    for neuron in range(no_brain_areas):
        temp_sample_list = sample_list.copy()
        temp_sample_list.remove(neuron)
        active_brain_areas = active_brain_areas + [[neuron] + random.sample(temp_sample_list, 2) for i in range(int(trials / no_brain_areas))]
    no_electrodes = projection_matrix.shape[0]
    EEG_Data = np.zeros((no_electrodes, time_steps, trials))
    activity = generate(seed, time_steps, trials)
    
    for trial in range(trials):
        
        noisy_activity = pink_noise(time_steps, noise_sources)
        noisy_activity = noisy_activity / np.linalg.norm(noisy_activity, ord = 'fro')
        activity_trial = activity[:,:,trial] / np.abs(activity[:,:,trial]).max()
        noisy_elements = np.random.choice(no_brain_areas, noise_sources, replace = False)
        
        EEG_activity = projection_matrix[:, active_brain_areas[trial]] @ activity_trial
        EEG_noisy = projection_matrix[:, noisy_elements] @ noisy_activity
        
        EEG_signal = sig_noise_ratio * EEG_activity + (1 - sig_noise_ratio) * EEG_noisy
        EEG_signal = EEG_signal / np.linalg.norm(EEG_signal, ord = "fro")
        
        EEG_channel_noise = np.random.normal(size = (no_electrodes, time_steps))
        EEG_channel_noise = EEG_channel_noise / np.linalg.norm(EEG_channel_noise, ord = "fro")
        
        EEG_Data[:,:,trial] = channel_noise_ratio * EEG_signal + (1 - channel_noise_ratio) * EEG_channel_noise
        activity[:,:,trial] = activity_trial
    
    active_brain_areas = np.array(active_brain_areas)
    shuffled_indexes = np.arange(trials).tolist()
    random.shuffle(shuffled_indexes)
    EEG_Data = EEG_Data[:,:,shuffled_indexes]
    activity = activity[:,:,shuffled_indexes]
    active_brain_areas = active_brain_areas[shuffled_indexes, :]
    data = (EEG_Data, active_brain_areas, activity)
    pickle.dump(data, open( "../../Data/EEG/data_" + str(sig_noise_ratio) + "_" + str(channel_noise_ratio) + "_" + str(noise_sources) + "_" + str(no_brain_areas) + ".pkl", "wb" ))
    
    if not only_save:
        return data

