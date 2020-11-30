# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:18:36 2020

@author: didie
"""
import random
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

#%%
def plot_line(traces, title = None):
    fig = []
    if title == None: title = str(random.random())
    for trace in traces:
        x = np.arange(len(trace))
        fig += [go.Scatter(x = x, y = trace, mode = "lines")]
    pyo.plot(fig, filename = title + '.html')
    
def plot_hyperparameters(results, goal_vars, hyper_params):
    for param in hyper_params:
        data = results[[param] + goal_vars]
        data = data.groupby(param).mean()
        fig = []
        for goal_var in goal_vars:
            fig += [go.Bar(name = goal_var, x = data.index.tolist(), y = data[goal_var])]
        pyo.plot(go.Figure(data = fig).update_layout(barmode='stack'), filename = param + '.jpg')
        
        
    
    
# #%%
# location_neurons = pickle.load(open("../../Data/location_neurons.pkl", "rb"))
# location_neurons = location_neurons[range(0, location_neurons.shape[0], 250),:]
# #%%
# neurons= np.arange(location_neurons.shape[0]).tolist()
# active = random.sample(neurons, 3)
# for neuron in active: neurons.remove(neuron)
# #%%
# noisy = random.sample(neurons, 150)
# for i in noisy: neurons.remove(i)
# idle = neurons
# #%%
# indexes = {"active": active, "noisy": noisy, "idle": idle}
# color = {"active": "red", "noisy": "pink", "idle": "grey"}
# opacity = {"active": 1, "noisy": 0.6, "idle": 0.3}
# #%%
# pyo.plot([go.Scatter3d(x = location_neurons[indexes[part],0],
#                       y = location_neurons[indexes[part],1],
#                       z = location_neurons[indexes[part],2],
#                       name = part,
#                       mode = 'markers',
#                       marker= dict(size = 5, color = color[part],
#                                    opacity = opacity[part]),
#                       ) for part in indexes.keys()])

# #%%
# parts = 100

# distances = np.zeros((parts, parts))

# cross_performances = np.zeros((parts, parts))

# cross_variance = np.zeros((parts, parts))
# for i in range(parts):
#     for j in range(parts):
#         distances[i,j] = ((location_neurons[i,:] - location_neurons[j, :])**2).mean(axis = 0)
#         cross_predict = NeuralNets[i](X[str(j)]["test"])
#         cross_performances[i,j] = np.float(((cross_predict - y[str(j)]["test"])**2).mean(axis = 0))
#         cross_variance[i,j] = cross_predict.std(axis = 0)

# #%%
# pyo.plot([go.Heatmap(z = np.exp(10*cross_variance), colorscale = 'Reds')])
# #%%
# pyo.plot([go.Heatmap(z = distances, colorscale = 'Reds')])
# #%%
# pyo.plot([go.Heatmap(z = np.log(cross_performances), colorscale = 'Reds')])
# #%%
# cross_performances
# top_cross =np.zeros((parts, parts))

# for i in range(parts):
#     top_cross[cross_performances[:, i].argmin(), i] = 1
#     #%%

# top_var =np.zeros((parts, parts))

# for i in range(parts):
#     top_var[cross_variance[:, i].argmax(), i] = 1
#       #%%
# pyo.plot([go.Heatmap(z = top_var, colorscale = 'Reds')])  
#   #%%
# pyo.plot([go.Heatmap(z = top_cross, colorscale = 'Reds')])  
# #%%
# pyo.plot([go.Scatter(x = np.arange(parts),
#            y = np.diag(cross_performances))])
# #%%
# pyo.plot([go.Scatter(x = np.arange(parts),
#            y = np.diag(cross_variance))])
# #%%
# pyo.plot([go.Scatter3d(x = location_neurons[:, 0],
#            y = location_neurons[:, 1],
#            z = location_neurons[:, 2],
#            mode = 'markers',
#            marker = dict(color = np.exp(20 *np.diag(cross_variance)), colorscale = 'Reds'))])