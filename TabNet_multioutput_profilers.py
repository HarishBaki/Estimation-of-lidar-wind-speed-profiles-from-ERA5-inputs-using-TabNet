#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime

# print date as date accessed
date_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Date accessed: {date_accessed}")


# In[2]:


import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, re, time, math, calendar, ast
import yaml

from pytorch_tabnet.tab_model import TabNetRegressor
import pickle
from pickle import dump, load
import joblib

import torch

from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

# import custom functions
sys.path.append('/')
from libraries import *
from plotters import *

#For reproducibility of the results, the following seeds should be selected 
from numpy.random import seed
randSeed = np.random.randint(1000)


# In[5]:

# === Input parameters ===
input_file = 'data/ERA5.nc'
input_variables = [
    "10ws", "100ws", "100alpha", "975ws", "950ws", "975wsgrad", "950wsgrad",
    "zust", "i10fg", "t2m", "skt", "stl1", "d2m", "msl", "blh", "cbh", "ishf", 
    "ie", "tcc", "lcc", "cape", "cin", "bld", "t_975", "t_950", "2mtempgrad", 
    "sktempgrad", "dewtempsprd", "975tempgrad", "950tempgrad", "sinHR", 
    "cosHR", "sinJDAY", "cosJDAY"
]
input_times_freq = 1 #ratio between the target times and input times

station_id = sys.argv[1]
Coeff_file = f'data/Profiler_Chebyshev_Coefficients/{station_id}.nc'
target_variables = [0,1,2,3,4]

train_dates_range = ('2018-01-01T00:00:00', '2019-12-31T23:00:00')
test_dates_range = ('2020-01-01T00:00:00', '2020-12-31T23:00:00')

experiment = f'ERA5_to_profilers'

tabnet_param_file = 'tabnet_params_8th_set.csv'
Ens = int(sys.argv[2])

model_output_dir = f'trained_models/{experiment}/{station_id}/Ens{Ens}'
os.system(f'mkdir -p {model_output_dir}')


# In[7]:


X_train, Y_train, X_valid, Y_valid = data_processing(input_file,Coeff_file,
                                                    input_times_freq,input_variables,target_variables,train_dates_range,station_id,val_arg=True)
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)


# In[8]:


X_test, Y_test = data_processing(input_file,Coeff_file,
                                input_times_freq,input_variables,target_variables,test_dates_range,station_id)
print(X_test.shape, Y_test.shape)


# In[9]:


# === load tabnet parameters ===
tabnet_params = pd.read_csv(tabnet_param_file)
n_d = int(tabnet_params['n_d'][Ens])
n_a = int(tabnet_params['n_a'][Ens])
n_steps = int(tabnet_params['n_steps'][Ens])
n_independent = int(tabnet_params['n_independent'][Ens])
n_shared = int(tabnet_params['n_shared'][Ens])
gamma = float(tabnet_params['gamma'][Ens])


# In[10]:


# === training the tabnet model ===#
tabReg   = TabNetRegressor(n_d = n_d, 
                                n_a = n_a, 
                                n_steps = n_steps,
                                n_independent = n_independent,
                                n_shared = n_shared,
                                gamma = gamma,
                                verbose=1,seed=randSeed, )


# In[11]:


tabReg.fit(X_train=X_train, y_train=Y_train,
                    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                    eval_name=['train', 'valid'],
                    max_epochs=250, batch_size=512,    #bSize_opt.item(), 
                    eval_metric=['rmse'], patience=10,  #mae, rmse
                    loss_fn = torch.nn.MSELoss())


# In[12]:


fSTR = f'{model_output_dir}/TabNet_HOLDOUT.pkl'
with open(fSTR, "wb") as f:
    dump(tabReg, f, pickle.HIGHEST_PROTOCOL)
print('dumped')


# In[13]:


# --- Plot loss curve and hexbin ---
fig = plt.figure(figsize=(18, 3), constrained_layout=True)
gs = fig.add_gridspec(1,6)

# Line plot for train and validation RMSE
ax = fig.add_subplot(gs[0])
ax.plot(tabReg.history['train_rmse'],'--', label='train')
ax.plot(tabReg.history['valid_rmse'],':', label='validation')
ax.set_title('Training and Validation RMSE')
ax.set_xlabel('Epochs')
ax.set_ylabel('RMSE')
ax.legend()

Y_pred = tabReg.predict(X_valid)
#Y_pred = min_max_scaler.inverse_transform(Y_pred)

for j,target_variable in enumerate(target_variables):
    hexbin_plotter(fig,gs[j+1],Y_valid[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel='Predicted')
fig.suptitle(f"n_d:{n_d}, n_a:{n_a}, n_steps:{n_steps}, n_independent:{n_independent}, n_shared:{n_shared}, gamma:{gamma}")

plt.savefig(f'{model_output_dir}/TabNet_HOLDOUT.png')
plt.close()


# In[ ]:




