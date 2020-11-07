# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:32:58 2020

@author: didie
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# #%%

# class DogsVSCats():
#     IMG_SIZE = 50
#     CATS = "PetImages/Cat"
#     DOGS = "PetImages/Dog"
#     TESTING = "PetImages/Testing"
#     LABELS = {CATS: 0, DOGS: 1}
#     X = []
#     y = []
    
#     def make_training_data(self):
#         for label in self.LABELS:
#             for f in tqdm(os.listdir(label)):
#                 if "jpg" in f:
#                     try:
#                         path = os.path.join(label, f)
#                         img = cv2.imread(path)
#                         img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
#                         self.X.append(np.array(img))
#                         self.y.append(np.eye(2)[self.LABELS[label]])  # do something like print(np.eye(2)[1]), just makes one_hot 
#                     except Exception as e:
#                         pass

#         rng_state = np.random.get_state()
#         np.random.shuffle(self.X)
#         np.random.set_state(rng_state)
#         np.random.shuffle(self.y)
        
#         self.X = torch.Tensor(self.X).view(-1, 50, 50, 3)
#         self.y = torch.Tensor(self.y)
# #%%
                        
# data = DogsVSCats()
# data.make_training_data()

#%%

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv3d(1, 32, 3) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv3d(32, 64, 2)
        
        x = torch.randn(10, 7, 6).view(-1, 1, 10, 7, 6)
        self._to_linear = None
        
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 252)
        self.fc2 =nn.Linear(252, 40)
        
        
    def convs(self, x):
        x  = F.max_pool3d(F.relu(self.conv1(x)), (2,2,1))
        x  = F.max_pool3d(F.relu(self.conv2(x)), (2,1,2))
        
        if self._to_linear == None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        return x
                          
net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()


#%%

VAL_PCT = 0.1
val_size = int(len(X_conv) * VAL_PCT)

X_nn = torch.Tensor(X_conv).view(-1, 10, 7, 6)
y_nn = torch.Tensor(y_cat)

train_X = X_nn[:-val_size]
train_y = y_nn[:-val_size]

test_X = X_nn[-val_size:]
test_y = y_nn[-val_size:]
#%%
BATCH_SIZE = 100

EPOCHS = 50

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_y), BATCH_SIZE)):
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 10, 7, 6)
        batch_y = train_y[i: i + BATCH_SIZE]
        
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
print(loss)
    #%%

acc = 0
with torch.no_grad():
    for i in tqdm(range(len(test_y))):
        real = test_y[i]
        pred = net(test_X[i].view(-1,1,10,7,6))[0]
        acc += 1 - sum((pred - real) **2) / 40
        
        

print("Accuracy: ",acc / len(test_y))
        
#%%
X_new = [X[i:(i+10)] for i in range(len(y))]
#%%
y_min =  min([min(row) for row in y])
y_max =  max([max(row) for row in y])
y_new = np.matrix(y)
y_new = (y_new - y_min )/(y_max - y_min)

