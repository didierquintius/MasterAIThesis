# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:48:20 2020

@author: didie
"""
import torch, os
import numpy as np
from copy import deepcopy
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def calc_loss(Net, X, y, loss_function):
    with torch.no_grad():
        loss = loss_function(Net(X), y)
    return np.float(loss)

class Validation():
    def __init__(self,max_val_amount, min_increment, loss_function, X, y):
        self.loss_function = loss_function
        self.val_counter = 0
        self.max_val_amount = max_val_amount
        self.min_val = np.inf
        self.min_Net = None
        self.val_losses = []
        self.X_val = X
        self.y_val = y
        self.min_increment = min_increment
        
    def update(self, Net):
        
        self.val_losses += [calc_loss(Net, self.X_val, self.y_val, self.loss_function)]
        if (self.min_val - self.val_losses[-1]  < self.min_increment):
            self.val_counter +=1
            if self.val_counter > self.max_val_amount:
                return Net, "Max val increment"
        else:
            self.min_Net = deepcopy(Net)
            self.min_val = self.val_losses[-1]
            self.val_counter = 0
        
        return Net, ""
    
    def reset(self):
        self.val_counter = 0
        self.min_val = np.inf
        self.min_Net = None
        self.val_losses = []
        
def createBatches(listi, n):
    return [listi[i:(i + n)] for i in range(0, len(listi), n)]

def updateNet(Net, X, y, batch, loss_function, optimizer):

    batch_X = X[batch, :]
    batch_y = y[batch, :]
    outputs = Net(batch_X)
    loss = loss_function(outputs, batch_y)
    loss.backward()
    optimizer.step()
    return loss

