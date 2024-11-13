import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, re, time, math, calendar

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# import custom functions
sys.path.append('/')
from libraries import *
from plotters import *

i = int(sys.argv[1])

output_dir = f'data/NOW23_profiles/zeroth_method'
os.makedirs(output_dir, exist_ok=True)

profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])

row = profiler_stations.iloc[i]
station_id = row['stid']
print(station_id)

target_file = f'{output_dir}/{station_id}.nc'
try:
    ds = xr.open_dataset(f'data/NOW23_profiles/{station_id}.nc').wind_speed
    ds = ds.resample(time='10min').mean(skipna=True)
    ds.to_netcdf(target_file)
    
except Exception as e:
    print(f'Error for station {station_id}: {e}')