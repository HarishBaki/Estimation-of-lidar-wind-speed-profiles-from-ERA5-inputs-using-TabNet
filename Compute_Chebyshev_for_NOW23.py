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

poly_order = 4
CPtype = 1

output_dir = f'data/NOW23_Chebyshev_Coefficients'
os.makedirs(output_dir, exist_ok=True)

profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])
for index, row in profiler_stations.iterrows():
    station_id = row['stid']
    print(station_id)

    target_file = f'{output_dir}/{station_id}.nc'
    if os.path.exists(target_file):
        print(f'Skipping station {station_id} as it already exists')
        continue

    try:
        # Load NOW23 data
        dfs = []
        for year in range(2000,2021):
            df = pd.read_csv(f'data/NOW23/{station_id}/{year}.csv',index_col=0)
            df.index = pd.to_datetime(df.index)
            dfs.append(df)
        combined_df = pd.concat(dfs,axis=0)

        Z = Z = np.array([10] + list(range(20, 301, 20)) + [400, 500])

        # Number of observations, time steps, and height levels
        n_time, n_height = combined_df.shape

        # Initialize Coeff array
        Coeff = np.zeros((n_time, poly_order+1))
        # Iterate over time steps
        U = combined_df.values
        stime = time.time()
        for t in range(n_time):
            Coeff[t, :] = Chebyshev_Coeff(Z, U[t,:],poly_order=poly_order, CPtype=CPtype, ref_H=ref_H)
        etime = time.time()
        print(f'Time elapsed: {etime-stime}s')

        coeff_da = xr.DataArray(Coeff, dims=['time', 'coeff'], coords={'time': combined_df.index.values, 'coeff': np.arange(poly_order+1)},name='Chebyshev_Coefficients')

        # Save the coefficients to a netCDF file
        
        coeff_da.to_netcdf(f'{target_file}.nc')
        print(f'Saved coefficients for station {station_id}')
    
    except Exception as e:
        print(f'Error for station {station_id}: {e}')
        continue

