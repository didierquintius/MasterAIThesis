# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:10:49 2020

@author: didie
"""
import torch.optim as optim
import torch.nn as nn
from NeuralNetwork import CNN
import torch


Net = CNN(50, [20, 40, 10], [3, 3], [2,1])
optimizer = optim.Adam(Net.parameters(), lr =  1e-4)
loss_function = nn.MSELoss()
val_counter = 0
#%%
for epoch in range(2):                
    for i in range(0, 10000, 10):
        Net.zero_grad()
        outputs = Net(torch.Tensor(predictions[i:(i + 10),:]).view(-1, 1, 50))
        loss = loss_function(outputs, torch.Tensor(source_matrix[i:(i + 10), 0]))
        loss.backward()
        optimizer.step()
        
#%%
prediction = Net(torch.Tensor(X[800:, :]).view(-1, 1, 100))
print(1 - np.sum((np.round(prediction.detach().numpy()) - y[800:, :])**2))