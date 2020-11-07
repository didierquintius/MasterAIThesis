# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:36:54 2020

@author: didie
"""

import os, pickle, random
import numpy as np
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from generate import generate

#%%
def pink_noise(n, m):
    n1= int(2 * np.ceil((n - 1)/2) + 1)
    scal = np.sqrt(1/np.arange(1,(n1-3)/2 + 1))
    ff=np.zeros((m, n1))
    ff[:, 1:int((n1-1)/2 )]=np.resize(scal,(m, scal.shape[0]))
    
    noise=np.fft.fft(np.random.normal(size = (n1,m)),axis = 0)
    noise=2*np.real(np.fft.ifft(np.transpose(np.multiply(noise, np.transpose(ff))), axis = 1))
    noise=noise[:, 0:n]
    return noise

def EEG_signal(time_steps,trials,elecs,sig_noise_ratio, channel_noise_ratio,
                noise_sources, seed = 0, reduction = 74):
    

    projection_matrix = pickle.load(open( "../Data/projection_matrix.pkl", "rb" ))
    projection_matrix = projection_matrix[:, range(0,projection_matrix.shape[1], reduction)]
    parts = projection_matrix.shape[1]
    if elecs == None:
        sample_list = np.arange(50).tolist()
        elecs = [[]]
        for i in parts:
            temp_sample_list = sample_list.copy().remove(i) 
            elecs = elecs + [[i] + random.sample(temp_sample_list, 2) for i in range(int(trials / parts))]
        
    EEG_Data = np.zeros((projection_matrix.shape[0], time_steps, trials))
    activity = generate(seed, time_steps, trials)

    for trial in tqdm(range(trials),mininterval = 2):
        elec = elecs[trial]
        noisy_activity = pink_noise(time_steps, noise_sources)
        noisy_activity = noisy_activity / np.linalg.norm(noisy_activity, ord = "fro")
        activity_trial = activity[:,:,trial]/ np.linalg.norm(activity[:,:,trial], ord = "fro")
        noisy_elements = np.random.choice(projection_matrix.shape[1], noise_sources,
                                          replace = False)
        
        EEG_activity = projection_matrix[:, elec] @ activity_trial
        EEG_noisy = projection_matrix[:, noisy_elements] @ noisy_activity
        
        EEG_signal = sig_noise_ratio * EEG_activity + (1 - sig_noise_ratio) * EEG_noisy
        EEG_signal = EEG_signal / np.linalg.norm(EEG_signal, ord = "fro")
        
        EEG_channel_noise = np.random.normal(size = (projection_matrix.shape[0],time_steps))
        EEG_channel_noise = EEG_channel_noise / np.linalg.norm(EEG_channel_noise, ord = "fro")
        
        EEG_Data[:,:,trial] = channel_noise_ratio * EEG_signal + (1 - channel_noise_ratio) * EEG_channel_noise
        activity[:,:,trial] = activity_trial
    return EEG_Data, elecs, activity

