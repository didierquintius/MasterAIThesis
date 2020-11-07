# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:33:13 2020

@author: didie
"""
#%%
import numpy.random as random
import numpy as np
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm 
import pickle
#%%
def circling(height_no):
    height = height_no[0]
    no = height_no[1]
    
    theta = np.linspace(0, 2*np.pi, no + 1)[:no]
    r = np.sqrt(1 - height**2)
    
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    x3 = np.ones(no) *height    
    return [(x1[i], x2[i], x3[i]) for i in range(no)]

def disty(x, y):
    dist = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)
    return dist

def gen_brain_points(no):
    
    phi = random.rand(no) * 2 * np.pi
    costheta = random.rand(no) 
    u = random.rand(no) * (1 - 1e-8)
    
    theta = np.arccos(costheta)
    r = np.sqrt(u)
    
    x = r * np.sin(theta) * np.cos( phi )
    y = r * np.sin(theta) * np.sin( phi )
    z = r * np.cos(theta)
    
    positions = [(x[i], y[i], z[i]) for i in range(no)]
    return positions, u

def generate_data(no , electrode_setup , TIME , active_region_size = 0.02, NOISE = 1):
    
    active_points = gen_brain_points(int(TIME / 50))[0]
    
    electrodes = []
    for level in electrode_setup: 
        electrodes += circling(level)
    
    coords = [np.array([])] * 3
    parts = len(active_points) - 1
    
    for i in range(parts):
        length = int((i + 1) / parts * (TIME - 1)) - int((i)/ parts  * (TIME - 1))
        for coord in range(3):
            lin_space = np.linspace(active_points[i][coord], 
                                    active_points[i + 1][coord], length + 1)
            coords[coord] = np.concatenate([coords[coord][:-1], lin_space])
            
    coords = [(coords[0][i], coords[1][i], coords[2][i]) for i in range(TIME)]
    
    pos = [(3,3), (2,3), (2,2), (3,2), (4,2), (4,3), (3,4), (2,4), (1,3), (1,2), (2,1), (3,1), (4,1), (5,2), (5,3), (4,4), (3,5), (2,5), (1,5), (1,4), (0,3), (0,2), (1,1), (2,0), (3,0), (4,0), (5,1), (6,2), (6,3), (5,4), (5,5), (4,5)]
    
    X = np.array([[[0]*6]*7] * (TIME + 20))
    
    activities = []
    
    positions, distances = gen_brain_points(no)
    
    for t in tqdm(range(TIME)):        
        active_region = multivariate_normal(mean = coords[t], cov =  active_region_size)
        activities_t = []

        for i in range(no):
            activity = active_region.pdf(positions[i])
            activities_t += [activity]            
            for j, p in enumerate(pos): 
                dist = disty(positions[i], electrodes[j])
                potential = 1 / dist * activity + np.random.normal() * NOISE
                arrival = int(dist * 12)
                try: X[arrival + t][p] += potential
                except: pass
        
        activities += [activities_t]
        
    return X, positions, activities, distances

#%%
electrode_setup = [(0.8, 6), (0.5, 10), (0.3, 16)]
TIME = 20000

X, pos, y, dist = generate_data(1000, electrode_setup , TIME, NOISE = 10)
#%%
X_conv = [(X[i:(i+20)] - X.min()) / (X.max() - X.min()) for i in range(len(y))]
#%%
def edit_target(y):
    cat = np.ones(len(pos), dtype = int)
    cat[[(p[0] > 0) & (p[1] < 0) for p in pos]] = 2
    cat[[(p[0] < 0) & (p[1] > 0) for p in pos]] = 3
    cat[[(p[0] < 0) & (p[1] < 0) for p in pos]] = 4
    for i in np.arange(0.1,1, 0.1): cat[dist > i] += 4
    
    y_cat = [[sum(np.array(neur)[cat == (category + 1)]) for category in range(40)] for neur in y]
    y_cat = np.matrix(y_cat)
    y_cat = (y_cat - y_cat.min())/(y_cat.max() - y_cat.min())
    
    return y_cat

y_cat = edit_target(y)
#%%
pickle.dump(X,open("X.pickle", "wb"))
pickle.dump(y,open("y.pickle", "wb"))




    