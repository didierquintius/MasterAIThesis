# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:24:25 2021

@author: didie
"""

import numpy as np
def findOptimalvalues(results):
    parameters = ['nodes_pred', 'nodes_Conv_clas', 'nodes_Dense_clas', 'kernel_size', 
                'strides', 'learning_rate_pred', 'learning_rate_clas', 'batch_sizes_pred',
                'batch_sizes_clas', 'val_treshold_pred', 'val_treshold_clas', 'max_val_amount_pred',
                'max_val_amount_clas', 'val_freq_pred', 'val_freq_clas', 'EPOCHS_pred', 'EPOCHS_clas',
                'trials', 'time_steps', 'brain_areas']

    centers = np.unique(results['brain_area'])
    centers_params = {}
    for center in centers:
        centers_params[center] = {}
        result = results[results['brain_area'] == center]
        for hyper_parameter in parameters:
            values = np.unique(result[hyper_parameter])
            value_score = []
            for value in values:
                value_result = result[result[hyper_parameter] == value][['true_positive_clas', 'true_negative_clas']]
                value_result['score'] = value_result['true_positive_clas'] + value_result['true_negative_clas']
                observations = value_result.shape[0]
                top_observation = int(95 / 100 * observations)
                sorted_observations = np.sort(value_result['score'])[-top_observation:]
                value_score += [sorted_observations.mean()]
            best_value = values[np.argmin(value_score)]
            centers_params[center][hyper_parameter] = best_value

    return centers_params
                