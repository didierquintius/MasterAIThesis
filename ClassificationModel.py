# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:30:40 2020

@author: didie
"""

#%%
import numpy as np
import torch, pickle, os, random
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork import CNN

#%%
def import_data(snr, cnr, noisy, neurons):
    
    file = "../../Data/NeuralNets"+ str(snr) +"_"+ str(cnr) +"_"+ str(noisy) +"_"+ str(neurons) +".pkl"
    NeuralNets  = pickle.load(open(file, "rb"))  
    
    data_file = "../../Data/EEG/data_"+ str(snr) +"_"+ str(cnr) +"_"+ str(noisy) +"_"+ str(neurons) +".pkl"
    EEG_data, sources, _ = pickle.load(open(data_file, "rb"))
    
    return NeuralNets, EEG_data, sources

def chunks(listi, n):
    return [listi[i:(i + n)] for i in range(0, len(listi), n)]
#%%
NeuralNets, EEG_data, sources = import_data(0.9, 0.9, 5, 10)
electrodes, timesteps, trials = EEG_data.shape
brainareas = len(NeuralNets)

#%%
EEG_data = EEG_data.reshape((electrodes, timesteps * trials), order = 'F')
EEG_data = torch.Tensor(EEG_data).transpose(0,1)
#%%
source_matrix = torch.Tensor(np.zeros((trials, timesteps)))
for trial, active_areas in enumerate(sources):
    source_matrix[trial, active_areas] = 1
#%%
Net = CNN(timesteps, [10, 15, 5], [3, 3], [2,2])
optimizer = optim.Adam(Net.parameters(), lr =  1e-4)
loss_function = nn.MSELoss()
val_counter = 0
#%%
def updateNet(Net, X, y, loss_function, optimizer):
    outputs = Net(X)
    loss = loss_function(outputs, y)
    loss.backward()
    optimizer.step()
    return loss

epochs = 2
no_blocks = np.round(trials / 3 - 1)
chunk_size = 100
losses = []
for brainarea in range(1):
    predictions = NeuralNets[brainarea](EEG_data).view((trials, timesteps))

    active = np.where(source_matrix[:, brainarea] == 1)[0].tolist()
    idle   = np.where(source_matrix[:, brainarea] == 0)[0].tolist()
    
    block_size = int(len(idle)/ no_blocks)   
    for epoch in range(epochs):
        for block in tqdm(range(no_blocks)):
            indexes = active + idle[block_size * block : block_size * (block + 1)]
            random.shuffle(indexes)
            prediction = predictions[indexes, :]
            prediction_label = source_matrix[indexes, brainarea]
            for chunk in chunks(np.arange(prediction.shape[0]), chunk_size):
                chunk_X = predictions[chunk, :].view(-1, 1, timesteps)
                chunk_y = torch.Tensor(prediction_label[chunk].view(-1,1))
                updateNet(Net, chunk_X, chunk_y, loss_function, optimizer)
