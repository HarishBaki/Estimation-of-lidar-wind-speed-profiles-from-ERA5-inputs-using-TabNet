import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, re, time, math, calendar

# import custom functions
sys.path.append('/')
from libraries import *

output_dir = f'data/NOW23_profiles'
os.makedirs(output_dir, exist_ok=True)

profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])
for index, row in profiler_stations.iterrows():
    station_id = row['stid']
    print(station_id)

    target_file = f'{output_dir}/{station_id}.nc'
    try:
        # Load NOW23 data
        dfs = []
        for year in range(2000,2021):
            df = pd.read_csv(f'data/NOW23/{station_id}/{year}.csv',index_col=0)
            df.index = pd.to_datetime(df.index)
            dfs.append(df)
        combined_df = pd.concat(dfs,axis=0)
        z = np.array([10] + list(range(20, 301, 20)) + [400, 500])
        profile_da = xr.DataArray(combined_df, dims=['time', 'levels'], coords={'time': combined_df.index.values, 'levels': z}, name='wind_speed')
        profile_da.to_netcdf(target_file)

    except Exception as e:
        print(f'Error for station {station_id}: {e}')
        continue