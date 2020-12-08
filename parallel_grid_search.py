# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:58:28 2020

@author: Quintius
"""

#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from Runv2 import runModel
from multiprocessing import Pool

if __name__ == '__main__':
    hi = Pool().map(runModel, range(4))


    
