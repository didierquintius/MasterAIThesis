# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 09:50:32 2020

% Code by Guido Nolte with minor modifications by Stefan Haufe
"""
import numpy as np

def eloreta(LL, gamma):
    nchan, ng = LL.shape
    u0 = np.eye(nchan)
    W = np.ones((1,ng))
    Winv = np.zeros((1, ng))
    winvkt = np.zeros((ng, nchan))
    kk = 0
    reldef = np.inf
    
    while (kk <= 20) & (reldef > 1e-6):
        kk += 1
        for i in range(ng):
            Winv[:, i] = 1 / W[:, i]
            
        for i in range(ng):
            winvkt[i, :] = Winv[:, i] * LL[:, i].T 
        
        kwinvkt = LL @ winvkt
        alpha = gamma * np.trace(kwinvkt) / nchan
        M = np.linalg.inv(kwinvkt + alpha * u0)
        for i in range(ng):
            Lloc = np.squeeze(LL[:, i])
            Wold = W.copy()
            W[:, i] = np.sqrt(Lloc.T  @ (M @ Lloc))
        
        reldef = np.linalg.norm(W - Wold) / np.linalg.norm(Wold)        
    ktm = LL.T @ M
    A = np.zeros((nchan, ng))
    for i in range(ng):
        A[:, i] = (Winv[:, i] * ktm[i, :]).T
    
    return A