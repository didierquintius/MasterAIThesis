# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:36:58 2020

@author: didie
"""

import matplotlib.pyplot as plt
hi = Net(X["train"])
#%%
plt.plot(range(75), hi[1325:1400,2].detach().numpy(),"k",
         range(75), y["train"][1325:1400,2],"r-")