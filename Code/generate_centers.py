# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:21:53 2021

@author: didie
"""

import pickle, os
import numpy as np
n_dipoles = 1000
n_centers = 12
coordinates = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\llocation_neurons.pkl', 'rb'))
coordinates = coordinates[range(0,coordinates.shape[0], int(coordinates.shape[0]/(n_dipoles - 1))),:][:n_dipoles,:]
#%%
distances = np.zeros((n_dipoles,n_dipoles))
def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2) 

for i in range(n_dipoles):
    for j in range(n_dipoles):
        distances[i, j] = distance(coordinates[i,:], coordinates[j,:])
#%%
centers = np.arange(0,n_dipoles,n_dipoles / n_centers).astype('int')
#%%
group = np.zeros((n_dipoles))
for i in range(20):
    for i in range(n_dipoles):
        group[i] = np.argmin(distances[i, centers])
    for i in range(n_centers):
        members = np.where(group == i)[0]
        center_group = coordinates[members].mean(axis = 0)
        average_coord = coordinates[members] - center_group
        average_distance = np.apply_along_axis(lambda x: x[0]**2 + x[1]**2 + x[2]**2, 1, average_coord)
        centers[i] = members[np.argmin(average_distance)]
    print(centers)








