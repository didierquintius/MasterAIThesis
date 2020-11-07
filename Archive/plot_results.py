# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:02:14 2020

@author: didie
"""

import plotly.graph_objs as go
import plotly.offline as pyo
import pandas as pd
import numpy as np
def plot_results(Net, X, y, key,title ="", partition ="test", plot_ranges = [range(100, 200)]):
    X = X[str(key)][partition]
    y = y[str(key)][partition]
    for plot_range in plot_ranges:
        pred = Net(X[plot_range, :]).detach().numpy().reshape(len(plot_range))
        real = y[plot_range, :].reshape(len(plot_range))

        print(pred.shape, real.shape)
        data = {"pred": pred, "real": real}
        
        fig = go.Figure(data = [{
            "x" : np.array(plot_range),
            "y" : data[data_type],
            "name": data_type,
            } for data_type in data.keys()],
            layout = go.Layout(title = title)) 
        
        pyo.plot(fig)
