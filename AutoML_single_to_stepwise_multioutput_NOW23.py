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
import os, sys, glob, re, time, math, calendar

import flaml
from flaml import AutoML
from multiprocessing import Process

import pickle
from pickle import dump, load
import joblib

# import custom functions
sys.path.append('/')
from libraries import *
from plotters import *

#For reproducibility of the results, the following seeds should be selected 
from numpy.random import seed
randSeed = np.random.randint(1000)

# Simulate passing arguments during debugging
if len(sys.argv) == 1:
    sys.argv = ['', ('PROF_OWEG'), 
                'Averaged_over_55th_to_5th_min', 
                ('2018-01-01T00:00:00', '2018-12-31T23:00:00'), 
                'not_segregated', 'not_transformed','r2',0, "1",0]    # for debugging
    print('Debugging mode: sys.argv set to ', sys.argv)

# stations can be passed as a list or a single string (for a single station) or a tuple of strings (for multiple stations)
# However, for debugging, we will pass a tuple of strings, so we need to convert it to a list.
if isinstance(sys.argv[1], tuple):
    station_ids = list(sys.argv[1])
elif isinstance(sys.argv[1], str):
    # Check if the string looks like a list or tuple
    if sys.argv[1].startswith('(') or sys.argv[1].startswith('['):
        station_ids = ast.literal_eval(sys.argv[1])  # For strings like "( 'PROF_CLYM', )" or "['PROF_CLYM']"
    else:
        station_ids = [sys.argv[1]]  # Treat it as a single string and convert it to a list
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
warm_start = int(sys.argv[9])

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

#test_station_ids = ('PROF_WANT','PROF_BRON','PROF_REDH','PROF_JORD')
test_station_ids = station_ids
test_dates_range = ('2020-01-01T00:00:00', '2020-12-31T23:00:00')

experiment = f'ERA5_to_NOW23'

tabnet_param_file = 'best_model_params.csv'

data_seed = randSeed
rng_data = np.random.default_rng(seed=data_seed)

#os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

if len(station_ids) == 1:
    station_id = station_ids[0]
    model_output_dir = f'trained_models/{experiment}/FLAML/{station_id}/{hourly_data_method}/{years_experiment}/{segregated}/{transformed}/{loss_function}/Ens{Ens}'
else:
    model_output_dir = f'trained_models/{experiment}/FLAML/{len(station_ids)}_stations/{hourly_data_method}/{years_experiment}/{segregated}/{transformed}/{loss_function}/Ens{Ens}'
os.system(f'mkdir -p {model_output_dir}')

# === Load the data ===
# Initialize empty lists to collect data for all stations
X_train_all, Y_train_all, X_valid_all, Y_valid_all = [], [], [], []
for station_id in station_ids:
    Coeff_file = f'data/NOW23_Chebyshev_Coefficients/{station_id}.nc'

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
    Coeff_file = f'data/NOW23_Chebyshev_Coefficients/{station_id}.nc'

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


# === Train the model ===

# === normalizing the training and validaiton data ---#
if transformed == 'transformed':
    min_max_scaler = preprocessing.MinMaxScaler().fit(Y_train)

    Y_train = min_max_scaler.transform(Y_train)
    Y_valid = min_max_scaler.transform(Y_valid)

    # --- save the normalizing function ---#
    joblib.dump(min_max_scaler, f'{model_output_dir}/min_max_scaler.joblib')
    print('min_max_scaler dumped')

automl_settings = {
    "time_budget": 7200,  # in seconds
    "metric": loss_function,
    "task": 'regression',
    "early_stop": True,
    "model_history": True, #A boolean of whether to keep the best model per estimator
    "retrain_full": True, #whether to retrain the selected model on the full training data
    "custom_hp": {
        "xgboost": {
            "tree_method": {
                "domain": "gpu_hist",       # Use GPU for tree construction
                "type": "fixed"
            },
            "predictor": {
                "domain": "gpu_predictor",  # Use GPU for prediction
                "type": "fixed"
            }
        }
    }
}

# === running the 9 process across 3 GPUs ===
gpu_devices = [0, 1, 2]

# Function to train base models
def train_base_model(target_variable, gpu_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print(f"Training base model for target {target_variable} on GPU {gpu_device}")

    automl = AutoML()
    X_tr, y_tr = X_train, Y_train[:, target_variable:target_variable+1]
    X_val, y_val = X_valid, Y_valid[:, target_variable:target_variable+1]

    model_path = f'{model_output_dir}/C{target_variable}.pkl'
    # --- for warm start ---#
    if warm_start and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            automl_prev_model = load(f)
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,starting_points=automl_prev_model.best_config_per_estimator, **automl_settings)
    else:
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val, **automl_settings)
    with open(model_path, "wb") as f:
        dump(automl, f)
    print(f"Base model {target_variable} saved to {model_path}")

# Function to train stepping stone models
def train_step_model(target_variable, gpu_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print(f"Training stepping stone model for target {target_variable} on GPU {gpu_device}")

    automl = AutoML()
    X_tr = np.hstack([X_train, Y_train[:, :target_variable]])
    y_tr = Y_train[:, target_variable:target_variable+1]
    X_val = np.hstack([X_valid, Y_valid[:, :target_variable]])
    y_val = Y_valid[:, target_variable:target_variable+1]

    model_path = f'{model_output_dir}/C{target_variable}_step{target_variable}.pkl'
    # --- for warm start ---#
    if warm_start and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            automl_prev_model = load(f)
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,starting_points=automl_prev_model.best_config_per_estimator, **automl_settings)
    else:
        automl.fit(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val, **automl_settings)
    with open(model_path, "wb") as f:
        dump(automl, f)
    print(f"Step model {target_variable} saved to {model_path}")

# Launch parallel training for base models
processes = []
idx = 0
for target_variable in ([0,1,2,3,4]):
    gpu_device = gpu_devices[idx % len(gpu_devices)]
    p = Process(target=train_base_model, args=(target_variable, gpu_device))
    p.start()
    processes.append(p)
    idx += 1

# Launch parallel training for stepping stone models
for target_variable in ([1, 2, 3, 4]):
    gpu_device = gpu_devices[idx % len(gpu_devices)]
    p = Process(target=train_step_model, args=(target_variable, gpu_device))
    p.start()
    processes.append(p)
    idx += 1

for p in processes:
    p.join()

print("Training completed.")


# === Plotting hexbins ===
fig = plt.figure(figsize=(15, 13), constrained_layout=True)
gs = fig.add_gridspec(4,len(target_variables))

# --- First row, with step 0 ---
Y_pred = []
for target_variable in target_variables:
    # load the respective model
    fSTR = f'{model_output_dir}/C{target_variable}.pkl'
    with open(fSTR, "rb") as f:
        model = load(f)
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1,1)
    Y_pred = np.hstack([Y_pred,y_pred]) if target_variable>0 else y_pred

if transformed == 'transformed':
    min_max_scaler = joblib.load(f'{model_output_dir}/min_max_scaler.joblib')
    Y_pred = min_max_scaler.inverse_transform(Y_pred)

for target_variable in (target_variables):
    ylabel = 'Single target\n Predicted' if target_variable == 0 else ''
    hexbin_plotter(fig,gs[0,target_variable],Y_test[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel=ylabel)

for target_variable in (target_variables):
    ax_qq = fig.add_subplot(gs[1,target_variable])
    ylabel = 'Single target\n Predicted' if target_variable == 0 else ''
    QQ_plotter(ax_qq,Y_test[:,target_variable],Y_pred[:,target_variable],title=f'Coefficient {target_variable}',label='',color='blue',xlabel='True',ylabel=ylabel,one_to_one=True)

# --- Second row, with step 1 to 4 ---
Y_pred = []
for target_variable in ([0,1,2,3,4]):
    # load the respective model
    fSTR = f'{model_output_dir}/C{target_variable}.pkl' if target_variable==0 else f'{model_output_dir}/C{target_variable}_step{target_variable}.pkl'
    with open(fSTR, "rb") as f:
        model = load(f)
    X_te = X_test if target_variable==0 else np.hstack([X_test,Y_pred])
    y_te = Y_test[:,target_variable:target_variable+1]
    y_pred = model.predict(X_te)
    y_pred = y_pred.reshape(-1,1)
    Y_pred = np.hstack([Y_pred,y_pred]) if target_variable>0 else y_pred

if transformed == 'transformed':
    min_max_scaler = joblib.load(f'{model_output_dir}/min_max_scaler.joblib')
    Y_pred = min_max_scaler.inverse_transform(Y_pred)

for target_variable in ([1,2,3,4]):
    ylabel = 'Steppingwise target\n Predicted' if target_variable == 1 else ''
    hexbin_plotter(fig,gs[2,target_variable],Y_test[:,target_variable],Y_pred[:,target_variable],f'Coefficient {target_variable}',text_arg=True, xlabel='True', ylabel=ylabel)

for target_variable in ([1,2,3,4]):
    ax_qq = fig.add_subplot(gs[3,target_variable])
    ylabel = 'Stepwise target\n Predicted' if target_variable == 0 else ''
    QQ_plotter(ax_qq,Y_test[:,target_variable],Y_pred[:,target_variable],title=f'Coefficient {target_variable}',label='',color='blue',xlabel='True',ylabel=ylabel,one_to_one=True)
plt.savefig(f'{model_output_dir}/hexbin_{warm_start}.png')
plt.close()