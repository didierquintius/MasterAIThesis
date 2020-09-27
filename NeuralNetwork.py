# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:35:20 2020

@author: didie
"""
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet1(nn.Module):
    
    def __init__(self, input_size, neurons, output = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, neurons[0])
        self.out = nn.Linear(neurons[-1], output)
        
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        
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
        x = self.out(x)
        
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
        x = self.out(x)
        
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
        x = self.out(x)
        
        return x

def NeuralNet(input_size, neurons,output):
    amount_of_layers = len(neurons)
    NeuralNets = {"1": NeuralNet1,
                  "2": NeuralNet2,
                  "3": NeuralNet3,
                  "4": NeuralNet4}
    
    return NeuralNets[str(amount_of_layers)](input_size, neurons, output)
    
