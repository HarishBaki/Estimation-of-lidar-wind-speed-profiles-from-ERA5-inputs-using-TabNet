#!/usr/bin/env python
# coding: utf-8

from datetime import datetime

# print date as date accessed
date_accessed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Date accessed: {date_accessed}")

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

# Simulate passing arguments during debugging
if len(sys.argv) == 1:
    sys.argv = ['', ('PROF_CLYM','PROF_OWEG','PROF_STAT','PROF_STON','PROF_QUEE','PROF_SUFF','PROF_BUFF','PROF_BELL','PROF_TUPP','PROF_CHAZ'), 
                'Averaged_over_55th_to_5th_min', 
                ('2018-01-01T00:00:00', '2018-12-31T23:00:00'), 
                'segregated', 'not_transformed','MSE_loss',9, "1"]    # for debugging
    print('Debugging mode: sys.argv set to ', sys.argv)

# stations can be passed as a list or a single string (for a single station) or a tuple of strings (for multiple stations)
# However, for debugging, we will pass a tuple of strings, so we need to convert it to a list.
if isinstance(sys.argv[1], tuple):
    station_ids = list(sys.argv[1])
elif isinstance(sys.argv[1], str):
    station_ids = ast.literal_eval(sys.argv[1])
else:
    station_ids = sys.argv[1]

hourly_data_method = sys.argv[2]

# train_dates_range can be passed as a tuple of strings or a string of a tuple of strings.
# However, for debugging, we will pass a tuple of strings, so we need to convert it to a list.
if isinstance(sys.argv[3], tuple):
    train_dates_range = list(sys.argv[3])
elif isinstance(sys.argv[3], str):
    train_dates_range = ast.literal_eval(sys.argv[3])
else:
    train_dates_range = sys.argv[3]

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

segregated = sys.argv[4]
transformed = sys.argv[5]
loss_function = sys.argv[6]
Ens = int(sys.argv[7])
gpu_device = sys.argv[8]

# === Input parameters ===
input_file = 'data/ERA5.nc'
input_variables = [
    "10ws", "100ws", "100alpha", "975ws", "950ws", "975wsgrad", "950wsgrad",
    "zust", "i10fg", "t2m", "skt", "stl1", "d2m", "msl", "blh", "ishf", 
    "ie", "tcc", "lcc", "cape", "bld", "t_975", "t_950", "2mtempgrad", 
    "sktempgrad", "dewtempsprd", "975tempgrad", "950tempgrad", "sinHR", 
    "cosHR", "sinJDAY", "cosJDAY"
]
input_times_freq = 1 #ratio between the target times and input times, 12 for NOW23 data

target_variables = [0,1,2,3,4]

test_station_ids = ('PROF_WANT','PROF_BRON','PROF_REDH','PROF_JORD')
test_dates_range = ('2019-01-01T00:00:00', '2020-12-31T23:00:00')

experiment = f'ERA5_to_profilers'

tabnet_param_file = 'best_model_params.csv'

data_seed = randSeed
rng_data = np.random.default_rng(seed=data_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

if len(station_ids) == 1:
    station_id = station_ids[0]
    model_output_dir = f'trained_models/{experiment}/{station_id}/{hourly_data_method}/{years_experiment}/{segregated}/{transformed}/{loss_function}/Ens{Ens}'
else:
    model_output_dir = f'trained_models/{experiment}/{len(station_ids)}_stations/{hourly_data_method}/{years_experiment}/{segregated}/{transformed}/{loss_function}/Ens{Ens}'
os.system(f'mkdir -p {model_output_dir}')

# === Load the data ===
# Initialize empty lists to collect data for all stations
X_train_all, Y_train_all, X_valid_all, Y_valid_all = [], [], [], []
for station_id in station_ids:
    Coeff_file = f'data/Profiler_Chebyshev_Coefficients_with_outliers/{hourly_data_method}/{station_id}.nc'

    if segregated == 'segregated':
        segregate_arg = True
    else:
        segregate_arg = None
    X_train, Y_train, X_valid, Y_valid,_,_ = data_processing_NYSP(input_file,Coeff_file,
                                                        input_times_freq,input_variables,target_variables,train_dates_range,station_id,val_arg=True, segregate_arg=segregate_arg,rng_data=rng_data)
    print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
    # Collect training and validation data for all stations
    X_train_all.append(X_train)
    Y_train_all.append(Y_train)
    X_valid_all.append(X_valid)
    Y_valid_all.append(Y_valid)

X_test_all, Y_test_all = [], []
for station_id in test_station_ids:
    Coeff_file = f'data/Profiler_Chebyshev_Coefficients_with_outliers/{hourly_data_method}/{station_id}.nc'

    if segregated == 'segregated':
        segregate_arg = True
    else:
        segregate_arg = None
    X_test, Y_test, _ = data_processing_NYSP(input_file,Coeff_file,
                                    input_times_freq,input_variables,target_variables,test_dates_range,station_id,val_arg=None, segregate_arg=segregate_arg)
    print(X_test.shape, Y_test.shape)
    # Collect testing data for all stations
    X_test_all.append(X_test)
    Y_test_all.append(Y_test)

# If there's more than one station, concatenate data from all stations
if len(station_ids) > 1:
    X_train = np.concatenate(X_train_all, axis=0)
    Y_train = np.concatenate(Y_train_all, axis=0)
    X_valid = np.concatenate(X_valid_all, axis=0)
    Y_valid = np.concatenate(Y_valid_all, axis=0)
else:
    # If only one station, no concatenation needed, reassign the variables from that station
    X_train = X_train_all[0]
    Y_train = Y_train_all[0]
    X_valid = X_valid_all[0]
    Y_valid = Y_valid_all[0]

if len(test_station_ids) > 1:
    X_test = np.concatenate(X_test_all, axis=0)
    Y_test = np.concatenate(Y_test_all, axis=0)
else:
    X_test = X_test_all[0]
    Y_test = Y_test_all[0]

# Print the shapes of the final datasets
print(f"Final X_train shape: {X_train.shape}")
print(f"Final Y_train shape: {Y_train.shape}")
print(f"Final X_valid shape: {X_valid.shape}")
print(f"Final Y_valid shape: {Y_valid.shape}")
print(f"Final X_test shape: {X_test.shape}")
print(f"Final Y_test shape: {Y_test.shape}")

# === normalizing the training and validaiton data ---#
if transformed == 'transformed':
    min_max_scaler = preprocessing.MinMaxScaler().fit(Y_train)

    Y_train = min_max_scaler.transform(Y_train)
    Y_valid = min_max_scaler.transform(Y_valid)

    # --- save the normalizing function ---#
    joblib.dump(min_max_scaler, f'{model_output_dir}/min_max_scaler.joblib')
    print('min_max_scaler dumped')

# === load tabnet parameters ===
tabnet_params = pd.read_csv(tabnet_param_file)
n_d = int(tabnet_params['n_d'][Ens])
n_a = int(tabnet_params['n_a'][Ens])
n_steps = int(tabnet_params['n_steps'][Ens])
n_independent = int(tabnet_params['n_independent'][Ens])
n_shared = int(tabnet_params['n_shared'][Ens])
gamma = float(tabnet_params['gamma'][Ens])

# === training the tabnet model ===#
tabReg   = TabNetRegressor(n_d = n_d, 
                                n_a = n_a, 
                                n_steps = n_steps,
                                n_independent = n_independent,
                                n_shared = n_shared,
                                gamma = gamma,
                                verbose=1,seed=randSeed, )

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

fSTR = f'{model_output_dir}/TabNet_HOLDOUT.pkl'
with open(fSTR, "wb") as f:
    dump(tabReg, f, pickle.HIGHEST_PROTOCOL)
print('dumped')

# --- Plot loss curve and hexbin ---
fig = plt.figure(figsize=(18, 3), constrained_layout=True)
gs = fig.add_gridspec(1,len(target_variables)+1)

# Line plot for train and validation RMSE
ax = fig.add_subplot(gs[0])
ax.plot(tabReg.history['train_rmse'],'--', label='train')
ax.plot(tabReg.history['valid_rmse'],':', label='validation')
ax.set_title('Training and Validation RMSE')
ax.set_xlabel('Epochs')
ax.set_ylabel('RMSE')
ax.legend()

Y_pred = tabReg.predict(X_test)
if transformed == 'transformed':
    Y_pred = min_max_scaler.inverse_transform(Y_pred)

for j,target_variable in enumerate(target_variables):
    hexbin_plotter(fig,gs[j+1],Y_test[:,j],Y_pred[:,j],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel='Predicted')
fig.suptitle(f"n_d:{n_d}, n_a:{n_a}, n_steps:{n_steps}, n_independent:{n_independent}, n_shared:{n_shared}, gamma:{gamma}")

plt.savefig(f'{model_output_dir}/TabNet_HOLDOUT.png')
plt.close()