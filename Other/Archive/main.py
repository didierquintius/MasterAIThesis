# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:30:23 2020

@author: Quintius
"""
from multiprocessing import Pool, freeze_support
from test import updateNet
from functools import partial
from torch import nn
import torch.optim as optim
from NeuralNetwork_functions import NeuralNet
import torch
import numpy as np


if __name__ == '__main__':
    print("hi")

    hi = Pool().map(updateNet, range(20))
    print("hi you")


