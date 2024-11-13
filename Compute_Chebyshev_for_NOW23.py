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

output_dir = f'data/NOW23_Chebyshev_Coefficients/zeroth_method'
os.makedirs(output_dir, exist_ok=True)

profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])
#for index, row in profiler_stations.iterrows():

i = int(sys.argv[1])
row = profiler_stations.iloc[i]
station_id = row['stid']
print(station_id)

target_file = f'{output_dir}/{station_id}.nc'

try:
    ds = xr.open_dataset(f'data/NOW23_profiles/zeroth_method/{station_id}.nc')
    data = ds.wind_speed.sel(levels=slice(10,500))
    Z = ds.levels.values

    target_file = f'{output_dir}/{station_id}.nc'
    # Number of observations, time steps, and height levels
    n_time, n_height = data.shape

    # Initialize Coeff array
    Coeff = np.zeros((n_time, poly_order+1))
    # Iterate over time steps
    U = data.values
    stime = time.time()
    for t in range(n_time):
        Coeff[t, :] = Chebyshev_Coeff(Z, U[t,:],poly_order=poly_order, CPtype=CPtype, ref_H=ref_H)
    etime = time.time()
    print(f'Time elapsed: {etime-stime}s')

    coeff_da = xr.DataArray(Coeff, dims=['time', 'coeff'], coords={'time': data.time.values, 'coeff': np.arange(poly_order+1)},name='Chebyshev_Coefficients')

    # Save the coefficients to a netCDF file
    
    coeff_da.to_netcdf(f'{target_file}')
    print(f'Saved coefficients for station {station_id}')

except Exception as e:
    print(f'Error for station {station_id}: {e}')