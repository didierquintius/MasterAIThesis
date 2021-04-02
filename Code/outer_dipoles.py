# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:16:03 2021

@author: didie
"""
import numpy as np
import pandas as pd
import random
class checkCoord():
    def __init__(self, a, b, c, coordinates):
        self.coordinates = coordinates
        a, b, c = coordinates[a,:], coordinates[b,:], coordinates[c,:]
        
        center = (a + b + c) / 4
        
        n1 = np.cross(a, b)
        n2 = np.cross(b, c)
        n3 = np.cross(a, c)
        n4 = np.cross(a - b, a - c)
        n4_base = sum(n4 * a)
        
        def direction(n, center = center):
            n = (2 * (sum(n * center) > 0) - 1) * n
            return n

        self.n1, self.n2, self.n3 = [direction(n) for n in [n1, n2, n3]]
        dir_n4 = 2 * (sum(n4 * center) - n4_base > 0) - 1
        self.n4 = dir_n4 * n4
        self.n4_base = dir_n4 * n4_base

    def check_dipole(self, d):
        d = self.coordinates[d,:]
        value = True
        for n in [self.n1, self.n2, self.n3]: value *= (sum(n * d) > 0)
        value *= (sum(self.n4 * d) - self.n4_base > 0)
        return value
    
#%%
import pickle, os
distances = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\dipole_distances.pkl', 'rb'))
short_distances = distances < 20
n_dipoles = distances.shape[0]
coordinates = pickle.load(open(os.environ['DATA'] + '\\MasterAIThesis\\llocation_neurons.pkl', 'rb'))
coordinates = coordinates[range(0,coordinates.shape[0], int(coordinates.shape[0]/(n_dipoles - 1))),:][:n_dipoles,:]
#%%
bac_df = pd.DataFrame(coordinates, columns = ["x", "y", "z"])
bac_df["section"] = bac_df.apply(lambda x: (x[0] > 0) + (x[1] > 0) * 2 + (x[2] > 0) * 4 , axis = 1)
bac_df["closeness_to_scalp"] = 0
#%%
def func_distance(x, coord):
    value = x[0]**2/np.abs(coord['x'])**2 + x[1]**2/np.abs(coord['y'])**2 + x[2]**2/np.abs(coord['z'])**2
    return value

maxCol = lambda x: max(np.abs(x.min()), np.abs(x.max()))

for section in range(8):
    maximum_coordinates_section = bac_df[bac_df['section'] == section].apply(maxCol,axis=0)
    bac_df.loc[bac_df['section'] == section,'closeness_to_scalp'] = bac_df.apply(lambda x: func_distance(x, maximum_coordinates_section), axis = 1)
#%%
outer = bac_df.index[bac_df['closeness_to_scalp'] > 0.7].tolist()
outer_size = len(outer)
#%%
while True:    
    a = random.sample(outer, 1)
    dipole_choice = np.where(short_distances[a,:])[1].tolist()
    if len(dipole_choice) < 3: continue
    a, b, c = random.sample(dipole_choice, 3)
    driehoek = checkCoord(a, b, c, coordinates)
    outer_new = []
    for dipole in outer:
        value = driehoek.check_dipole(dipole)
        if value: short_distances[:, dipole] = False
        else: outer_new += [dipole]
    outer = outer_new
    print(len(outer))
#%%
import plotly.offline as pyo
import plotly.graph_objs as go

pyo.plot([go.Scatter3d(x = bac_df.loc[plot_dipoles, 'x'], y =  bac_df.loc[plot_dipoles, 'y'], z = bac_df.loc[plot_dipoles, 'z'],
                      mode = 'markers', marker= dict(size = 5, color = 'red')),
          go.Scatter3d(x = bac_df.loc[:, 'x'], y =  bac_df.loc[:, 'y'], z = bac_df.loc[:, 'z'],
                      mode = 'markers', marker_symbol = 'circle-open',
                      marker= dict(size = 5, color = 'black'))])

#%%
outer_distances = distances[:, outer]
distance_to_scalp = outer_distances.min(axis = 0)
#%%
pickle.dump(distance_to_scalp, open(os.environ['DATA'] + '\\MasterAIThesis\\distance_to_scalp.pkl', 'wb'))

    
    
    
        
