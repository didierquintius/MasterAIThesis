
import os, pickle, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import loadmat

def transform_data(EEG_data, lag):
    elecs, time_steps, trials = EEG_data.shape
    transformed_EEG_data = np.zeros((elecs * lag, time_steps - lag + 1, trials))
    for trial in range(trials):
        for time_step in range(time_steps - lag + 1):
            segment = EEG_data[:, time_step:(time_step + lag), trial]
            transformed_EEG_data[:, time_step, trial] = segment.reshape(elecs * lag)
     
    return transformed_EEG_data

def split_data(X, y, validation_perc = 0.1, test_perc = 0.1):
    
    elecs, time_steps, trials = X.shape
    
    train_ind = int((1 - validation_perc - test_perc) * trials)
    val_ind = int((1 - test_perc) * trials)
    
    X_train = torch.Tensor(X[:,:,:train_ind].copy()).view(elecs, time_steps, train_ind)
    X_val = torch.Tensor(X[:,:,train_ind:val_ind].copy()).view(elecs, time_steps, val_ind - train_ind)
    X_test = torch.Tensor(X[:,:,val_ind:trials].copy()).view(elecs, time_steps, trials - val_ind)
    
    y_train = torch.Tensor(y[:,:,:train_ind].T.copy()).view(3, time_steps, train_ind)
    y_val = torch.Tensor(y[:,:,train_ind:val_ind].T.copy()).view(3, time_steps, val_ind - train_ind)
    y_test = torch.Tensor(y[:,:,val_ind:trials].T.copy()).view(3, time_steps, trials - val_ind)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def do_val_check(net, X, y, lag, plot = False):
    loss = 0
    with torch.no_grad():
            for i in range(y.shape[2]):
                real = y[:,:,i].transpose(0,1).view(-1, 3) * 100
                pred = net(real)
                if plot and (np.mod(i, 30) == 0):
                    for k in range(3):
                        plt.plot(range(200), real[0:200, k],  "r-", 
                                 range(200), pred[0:200,k], "b-")
                        plt.show()
                loss += loss_function(pred, X[:,:,i].transpose(0,1).view(-1, 108 * lag))
                
    return np.float(loss)

class Net(nn.Module):
    def __init__(self,input_size, l1, l2, out):
        super().__init__() # just run the init of parent class (nn.Module)
        self.fc1 = nn.Linear(input_size,l1)
        self.fcout = nn.Linear(l1, out)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fcout(x)
        return x

#%%
EEG_data, labels, source = pickle.load(open("../Data/EEG/data_0.9_0.9_500.pkl", "rb"))
lag = 1
X = transform_data(EEG_data[:,:,:100], lag)

y = source[:, 0:(source.shape[1] - lag + 1), :100]
del EEG_data, source 

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)  

#%%                     
net = Net(3, 6, 12, 108 * lag)
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
loss_function = nn.MSELoss()

#%%
BATCH_SIZE = 25
val_stop = 20
EPOCHS = 1
min_val = np.inf
val_check = 1000
val_counter = 0
for epoch in range(EPOCHS):
    for i in tqdm(range(X_train.shape[2])):
        # if val_check > min_val: 
        #     val_counter += 1
        #     if val_counter > val_stop: 
        #         print('Validation STOP')
        #         break          
        # else: 
        #     min_val = val_check
            # val_counter = 0
        
        for j in range(0, X_train.shape[1], BATCH_SIZE):
            batch_X = X_train[:, j : (j + BATCH_SIZE),i].transpose(0,1).view(-1, 108 * lag)
            batch_y = y_train[:, j : (j + BATCH_SIZE),i].transpose(0,1).view(-1, 3) * 100
            
            net.zero_grad()
            outputs = net(batch_y)
            loss = loss_function(outputs, batch_X)
            loss.backward()
            optimizer.step()
        # val_check = do_val_check(net, X_val, y_val, lag)
print()
print("Loss: " + str(np.float(loss)))
print("Accuracy: ", do_val_check(net, X_test, y_test, lag, plot = True))


