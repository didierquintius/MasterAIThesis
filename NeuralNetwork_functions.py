# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:35:20 2020

@author: didie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet1(nn.Module):
    
    def __init__(self, input_size, neurons, output = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, neurons[0])
        self.out = nn.Linear(neurons[-1], output)
        
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.out(x))
        
        return x
    
class NeuralNet2(nn.Module):
    
    def __init__(self, input_size, neurons, output = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.out = nn.Linear(neurons[1], output)
        
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))        
        return x

class NeuralNet3(nn.Module):
    
    def __init__(self, input_size, neurons, output = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])
        self.out = nn.Linear(neurons[-1], output)
        
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.out(x))
        
        return x
    
class NeuralNet4(nn.Module):
    
    def __init__(self, input_size, neurons, output = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])
        self.fc4 = nn.Linear(neurons[2], neurons[3])
        self.out = nn.Linear(neurons[-1], output)
        
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.out(x))
        
        return x

def NeuralNet(input_size, neurons,output):
    amount_of_layers = len(neurons)
    NeuralNets = {"1": NeuralNet1,
                  "2": NeuralNet2,
                  "3": NeuralNet3,
                  "4": NeuralNet4}
    
    return NeuralNets[str(amount_of_layers)](input_size, neurons, output)
#%%
def output_size(input_size, kernel_size, stride):
    return int((input_size - kernel_size + stride) / stride)
    
class CNN2(nn.Module):  
    
    def __init__(self, input_size, layers, kernel_sizes, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(1, layers[0], kernel_sizes[0], stride[0])
        self.conv2 = nn.Conv1d(layers[0], layers[1], kernel_sizes[1], stride[1])
        cnn_output = output_size(input_size, kernel_sizes[0], stride[0])
        self.cnn_output = output_size(cnn_output, kernel_sizes[1], stride[1]) * layers[1]
        self.fc1 = nn.Linear(self.cnn_output, layers[2])
        self.fc2 = nn.Linear(layers[2], 1)
        
    def forward(self, x):
        x = x.detach()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.cnn_output)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
class CNN(nn.Module):  
    
    def __init__(self, input_size, layers, kernel_sizes, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(1, layers[0], kernel_sizes[0], stride[0])
        self.cnn_output = output_size(input_size, kernel_sizes[0], stride[0]) * layers[0]
        self.fc1 = nn.Linear(self.cnn_output, layers[1])
        self.fc2 = nn.Linear(layers[1], 1)
        
    def forward(self, x):
        x = x.detach()
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.cnn_output)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
        
        

