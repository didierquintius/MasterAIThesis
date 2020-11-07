# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:15:59 2020

@author: didie
"""

resulty = Net(X)
plt.plot(range(100),y["0"]["test"][300:400, 0].numpy(),"k",range(100), resulty[300:400].detach().numpy(),"b")