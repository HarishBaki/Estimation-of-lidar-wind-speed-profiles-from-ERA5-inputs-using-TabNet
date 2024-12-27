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
randSeed = 42


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
input_times_freq = 1 #ratio between the target times and input times, 12 for NOW23 data

#sys.argv = ['', 'PROF_QUEE','Averaged_over_55th_to_5th_min', 'segregated', 'not_transformed','Kho_loss_on_profile',0,  42, 0, 80, "1"]    # for debugging
station_id = sys.argv[1]
hourly_data_method = sys.argv[2]
Coeff_file = f'data/Profiler_Chebyshev_Coefficients_with_outliers/{hourly_data_method}/{station_id}.nc'
target_variables = [0,1,2,3,4]

train_dates_range = ('2021-01-01T00:00:00', '2023-12-31T23:00:00')

# Extract years from the date range
start_date = datetime.fromisoformat(train_dates_range[0])
end_date = datetime.fromisoformat(train_dates_range[1])
# Get the years
start_year = start_date.year
end_year = end_date.year
# Format the folder name
if start_year == end_year:
    years_experiment = f"{start_year}"
else:
    years_experiment = f"{start_year}_to_{end_year}"

experiment = f'ERA5_to_profilers'

segregated = sys.argv[3]
transformed = sys.argv[4]
loss_function = sys.argv[5]
Ens = int(sys.argv[6])
data_seed = int(sys.argv[7])
trial = int(sys.argv[8])
hp_seed = int(sys.argv[9])
gpu_device = sys.argv[10]

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

n_d = np.array([4, 8, 16])
n_steps = np.array([3, 4, 5])
n_independent = np.array([1, 2, 3, 4, 5])
n_shared = np.array([1, 2, 3, 4, 5])
gamma = np.array([1.1, 1.2, 1.3, 1.4])

rng_data = np.random.default_rng(seed=data_seed)
rng_hp = np.random.default_rng(seed=hp_seed)

start_index = rng_data.choice(1000)

n_d_opt = np.squeeze(rng_hp.choice(n_d,1))
n_a_opt = n_d_opt
n_steps_opt = np.squeeze(rng_hp.choice(n_steps,1))
n_independent_opt = np.squeeze(rng_hp.choice(n_independent,1))
n_shared_opt = np.squeeze(rng_hp.choice(n_shared,1))
gamma_opt = np.squeeze(rng_hp.choice(gamma,1))

print("trial = {:d} start_index = {:d} n_d = {:d} n_steps = {:d} n_independent = {:d}"
        " n_shared = {:d} gamma = {:f}".format(trial,start_index, n_d_opt.item(),n_steps_opt.item(),n_independent_opt.item(),n_shared_opt.item(),gamma_opt.item()))


model_output_dir = f'trained_models/{experiment}/{station_id}/{hourly_data_method}/{years_experiment}/{segregated}/{transformed}/{loss_function}/Ens{Ens}/trial{trial}'
os.system(f'mkdir -p {model_output_dir}')

# In[7]:
if segregated == 'segregated':
    segregate_arg = True
else:
    segregate_arg = None
X_train, Y_train, X_valid, Y_valid = data_processing_NYSP(input_file,Coeff_file,
            input_times_freq,input_variables,target_variables,train_dates_range,station_id,val_arg=True, segregate_arg=segregate_arg,rng_data=rng_data)
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

# === normalizing the training and validaiton data ---#
if transformed == 'transformed':
    min_max_scaler = preprocessing.MinMaxScaler().fit(Y_train)

    Y_train = min_max_scaler.transform(Y_train)
    Y_valid = min_max_scaler.transform(Y_valid)

    # --- save the normalizing function ---#
    joblib.dump(min_max_scaler, f'{model_output_dir}/min_max_scaler.joblib')
    print('min_max_scaler dumped')

# In[9]:
# === training the tabnet model ===#
tabReg   = TabNetRegressor(n_d = n_d_opt.item(), 
                                n_a = n_a_opt.item(), 
                                n_steps = n_steps_opt.item(),
                                n_independent = n_independent_opt.item(),
                                n_shared = n_shared_opt.item(),
                                gamma = gamma_opt.item(),
                                verbose=1,seed=randSeed, )


# In[11]:

if loss_function == 'L1_loss':
    loss_fn = L1_loss
elif loss_function == 'MSE_loss':
    loss_fn = MSE_loss
elif loss_function == 'weighted_MSE_loss':
    loss_fn = weighted_MSE_loss
elif loss_function == 'focal_MSE_loss':
    loss_fn = focal_MSE_loss
elif loss_function == 'profiler_loss':
    loss_fn = profiler_loss
elif loss_function == 'Kho_loss':
    loss_fn = Kho_loss
elif loss_function == 'Kho_loss_on_profile':
    loss_fn = Kho_loss_on_profile
else:
    print('Unknown loss function')
    sys.exit(1)

tabReg.fit(X_train=X_train, y_train=Y_train,
                    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
                    eval_name=['train', 'valid'],
                    max_epochs=250, batch_size=512,    #bSize_opt.item(), 
                    eval_metric=['rmse'], patience=10,  #mae, rmse
                    loss_fn = loss_fn) #weighted_mse_loss, #focal_mse_loss


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
if transformed == 'transformed':
    Y_pred = min_max_scaler.inverse_transform(Y_pred)

for j,target_variable in enumerate(target_variables):
    hexbin_plotter(fig,gs[j+1],Y_valid[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel='Predicted')
fig.suptitle(f"n_d:{n_d_opt.item()}, n_a:{n_a_opt.item()}, n_steps:{n_steps_opt.item()}, n_independent:{n_independent_opt.item()}, n_shared:{n_shared_opt.item()}, gamma:{gamma_opt.item()}")

plt.savefig(f'{model_output_dir}/TabNet_HOLDOUT.png')
plt.close()