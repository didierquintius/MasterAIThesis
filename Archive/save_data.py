# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:42:47 2020

@author: didie
"""



from forward_problem import EEG_signal
#%%
a, b, c, d = EEG_signal(time_steps = 10, trials = 100, sig_noise_ratio = 0.9,
                  channel_noise_ratio=0.9, no_brain_areas= 100)

