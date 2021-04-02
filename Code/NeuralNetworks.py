# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:02:34 2021

@author: didie
"""

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

def NN(n_input, n_nodes, n_output):
    model = Sequential()
    model.add(Dense(n_nodes, input_dim=n_input, activation='relu'))
    model.add(Dense(n_output, activation = 'tanh'))
    return model

def CNN(n_input, nodes, dense_nodes, kernel, stride, output):
# Build the model.
    model = Sequential()
    model.add(Conv1D(nodes, ([kernel]), strides = ([stride]), input_shape = (n_input, 1)))
    model.add(Flatten())
    model.add(Dense(dense_nodes, activation = 'relu'))
    model.add(Dense(output, activation='sigmoid'))

    return model