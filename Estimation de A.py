#!/usr/bin/env python
# coding: utf-8

# # Matrix Approximation

import torch
import torch.nn as nn
from torch import optim
#from torchdiffeq import odeint_adjoint as odeint
from scipy.linalg import expm
from torchdiffeq import odeint
import numpy as np
import csv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


# ### Defining essential functions


def dydt(y, t, A):
    return torch.mm(y,A)

def phi_A(y,A):
    return odeint(lambda t,x : dydt(x,t,A), y, torch.tensor([0., 1.]))[1]


# ### Neural ODE structure

# In[4]:


class ODEFunc(torch.nn.Module):
    def __init__(self, A):
        super(ODEFunc, self).__init__()
        self.A = torch.nn.Parameter(torch.tensor(A))
        
    def forward(self, t, y):
        return dydt(y, t, self.A)


# In[5]:


class NeuralODE(torch.nn.Module):
    def __init__(self, A_init):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc(A_init)
        self.dim=len(A_init)
        self.hidden_layer = torch.nn.Linear(self.dim, 100)
        self.output_layer = torch.nn.Linear(100, self.dim*self.dim)
        
    def forward(self, y):
        y = self.hidden_layer(y)
        y = torch.relu(y)
        y = self.output_layer(y)
        return y
    
    def get_A(self):
        return self.func.A


# ## Training



def train_model(model, x_data,y_data,epochs=300, lr=0.05):
    training_loss=[]
    y_pred_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = odeint(model.func, x_data, torch.tensor([0., 1.]), method='dopri5')[1]
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            #print(f"Epoch {epoch}, Training Loss: {loss:.4f}")
            predict=torch.diag(model.get_A())
            #print(predict)
            training_loss.append(loss.detach().numpy().item())
            y_pred_list.append(predict.detach().numpy())
    return training_loss,y_pred_list


def train(x_data,n_samples,stddev,mean,dim):
    print("\n")
    print("n_samples :",str(n_samples))
    print("déviation standard : ",str(stddev))

    diag = torch.from_numpy(np.random.normal(loc=mean, scale=stddev, size=dim).astype(np.float32))
    A_true = torch.diag(diag)
    #print("matrice à viser : " , diag)

    y_data = phi_A(x_data, A_true)

    neural_ode = NeuralODE(np.eye(dim).astype(np.float32))
    training_loss,y_pred_list = train_model(neural_ode, x_data, y_data)
    A_estimated = neural_ode.get_A()
    frob_losses=np.linalg.norm((A_true-A_estimated).detach().numpy())
    print("norme de frobenius", str(frob_losses),"\n",flush=True)
    key = str(stddev)
    data[key] = {'training_loss': training_loss, 'A_true': diag,'y_pred':y_pred_list,'frob':frob_losses}
    


# ### Parameters

# In[65]:


dim = 10 #dimension of the matrix
#mean = 1 #mean of the diagonal values
#n_samples=8 #number of samples
stddevs=[i/10 for i in range(1,11)] #standard deviations from 0.1 to 1

file_path = 'data'


#gaussian distribution for the x_i



""" adding noise to y
stddev=0.05
diag = torch.from_numpy(np.random.normal(loc=mean, scale=stddev, size=dim).astype(np.float32))
A_true = torch.diag(diag)
y_data = phi_A(x_data, A_true)
noise_stddev = stddev/10  
noise = noise_stddev * torch.randn_like(y_data) 
y_noisy = y_data + noise
"""





data = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=10, help='dimension of the matrix')
    parser.add_argument('--mean', type=float, default=1, help='mean of the diagonal values')
    parser.add_argument('--n_samples_list', type=int, nargs='+', default=[5,10,20,50,100], help='list of number of samples')
    args = parser.parse_args()
    
    dim = args.dim
    mean = args.mean
    n_samples_list=args.n_samples_list
    
    for n_samples in n_samples_list:
        x_data = torch.randn(n_samples, dim)
        for stddev in tqdm(stddevs):
            train(x_data,n_samples,stddev,mean,dim)

        np.save(file_path+'_'+str(n_samples), data)



"""
#loading the data trained
n_samples=100
if os.path.exists(file_path+'_'+str(n_samples)+'.npy'):
    data = np.load(file_path+'_'+str(n_samples)+'.npy', allow_pickle=True).item() 
"""
