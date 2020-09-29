# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:10:49 2020

@author: didie
"""
import torch.optim as optim
import torch.nn as nn
from NeuralNetwork import CNN
import torch


Net = CNN(100, [20, 40, 10], [10, 3], [5,1])
optimizer = optim.Adam(Net.parameters(), lr =  1e-4)
loss_function = nn.MSELoss()
val_counter = 0
#%%
for epoch in range(10):                
    for i in range(0, 800, 10):
        Net.zero_grad()
        outputs = Net(torch.Tensor(X[i:(i + 10),:]).view(-1, 1, 100))
        loss = loss_function(outputs, torch.Tensor(y[i:(i +10), :]))
        loss.backward()
        optimizer.step()
        
#%%
prediction = Net(torch.Tensor(X[800:, :]).view(-1, 1, 100))
print(1 - np.sum((np.round(prediction.detach().numpy()) - y[800:, :])**2))