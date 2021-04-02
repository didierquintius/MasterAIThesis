# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:43:57 2021

@author: didie
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from OptimalGridSearchResults import findOptimalvalues
from generate_centers import calculate_centers
from BrainAreaFunctions import train_brain_area
import numpy as np
import pandas as pd

from tqdm import tqdm
import pickle

def trainMLModel(results):
    optimal_hyper_params = findOptimalvalues(results)
    centers = np.unique(results['brain_area']).tolist()
    _, dipole_centers = calculate_centers(centers = centers)
    n_dipoles = 1000
    results = {'NNs':{}, 'CNNs':{}, 'MSEpred':{}, 'MSEclas':{}, 'TA':{}, 
               'TN':{}, 'TI':{}, 'accuracy': {}}
    
    for dipole in tqdm(range(n_dipoles)):
        dipole_center = centers[dipole_centers[dipole]]
        results['MSEpred'][dipole],  results['MSEclas'][dipole],  results['TA'][dipole], results['TN'][dipole],  results['TI'][dipole],  results['accuracy'][dipole], _ , _ ,  results['NNs'][dipole],  results['CNNs'][dipole]= train_brain_area(dipole, optimal_hyper_params[dipole_center], add_networks=True)

        
    pickle.dump(results, open('./TrainResults.pkl','wb'))
 #%%   
if __name__ == '__main__':
    results = pd.read_csv("C:/Users/didie/Documents/MasterAIThesis/Code/FinalResults/CodeServerGridsearch/results.csv")
    results['accuracy'] = results['true_positive_clas'] / 3 + results['true_negative_clas'] * 2 / 3  
    #trainMLModel(results)