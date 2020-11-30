# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:17:21 2020

@author: Quintius
"""
import torch
import numpy as np
from DataSplit_functions import setNNFormat
def FeatureImportance(NeuralNets, elecs = 108):
    
    partial_data = torch.eye(elecs) * 0.99 + 0.01
    importance = torch.zeros((len(NeuralNets), elecs))
    for brain_area in range(len(NeuralNets)):
        importance[brain_area, :] = NeuralNets[brain_area](partial_data).view(-1, elecs)
    
    return importance
        
    