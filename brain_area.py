# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:49:51 2021

@author: Quintius
"""

import pickle, os
import numpy as np
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
brain_coord = pickle.load(open( "../Data/brain_area_coordinates.pkl", "rb" ))
#%%
brain_coord = brain_coord[np.arange(0, brain_coord.shape[0], 74), :] 
bac_df = pd.DataFrame(brain_coord, columns = ["x", "y", "z"])
#%%
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
import plotly.offline as pyo
import plotly.graph_objs as go
sub = bac_df[bac_df['closeness_to_scalp'] > 1]
pyo.plot([go.Scatter3d(x = sub['x'], y =  sub['y'], z = sub['z'],
                      mode = 'markers')])
