import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, re, time, math, calendar

import dask
import dask.distributed as dd

profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])

# --- Extracting surface level variables ---#
pars = ['u10', 'v10', 'u100', 'v100','zust','i10fg',
            't2m','skt','stl1','d2m','msl','blh','cbh',
            'ishf','ie','tcc','lcc','cape','cin','bld']

for par in (pars):
    # create a folder with name par inside data_dir
    par_dir = f'data/ERA5_variables/{par}'
    os.makedirs(par_dir, exist_ok=True)

    # --- combining all years data ---#
    ds = xr.open_mfdataset(f'{par_dir}/*.nc', 
                                parallel=True)
    file_path = f'data/ERA5_variables/{par}.nc'
    if os.path.exists(file_path):
        os.remove(file_path)
    ds.to_netcdf(file_path)
    print(par)

# --- Extracting pressure level variables ---#
lvls = [1000,975,950]
pars = ['u','v','t']
for par in pars:
    for level in lvls:
        # --- combining all years data ---#
        ds = xr.open_mfdataset(f'{par_dir}/*.nc',
                                    parallel=True)
        file_path = f'data/ERA5_variables/{par}_{level}.nc'
        if os.path.exists(file_path):
            os.remove(file_path)
        ds.to_netcdf(file_path)
        print(par)