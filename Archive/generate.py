# -*- coding: utf-8 -*-
"""
Created on Thu Maactivity 28 23:18:24 2020

@author: didie
"""
import numpy as np

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
