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

profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])
for index, row in profiler_stations.iterrows():
    lon = row['lon [degrees]']
    lat = row['lat [degrees]']
    station_id = row['stid']
    print(index, lat, lon, station_id)

    try:
        # Load profiler data
        ds = xr.open_dataset(f'data/NYSM_standard_and_profiler_combined_wind_speed/{station_id}.nc')
        data = ds.wind_speed.sel(range=slice(10,500))
        Z = ds.range.values

        # Define the number of required non-missing points for intermediate range
        intermediate_non_missing_points = 4

        # Masks to ensure non-missing points in each segment
        first_segment_mask = data.sel(range=slice(100, 200)).notnull().sum(dim='range') >= 2
        last_segment_mask = data.sel(range=slice(400, 500)).notnull().sum(dim='range') >= 2
        intermediate_segment_mask = data.sel(range=slice(225, 375)).notnull().sum(dim='range') >= intermediate_non_missing_points

        # Combine masks to select data meeting all conditions
        conditional_non_missing_mask = first_segment_mask & last_segment_mask & intermediate_segment_mask
        filtered_data = data.where(conditional_non_missing_mask, drop=True)

        # Number of observations, time steps, and height levels
        n_height,n_time = filtered_data.shape

        # Initialize Coeff array
        Coeff = np.zeros((n_time, poly_order+1))
        # Iterate over time steps
        U = filtered_data.values
        stime = time.time()
        for t in range(n_time):
            Coeff[t, :] = Chebyshev_Coeff(Z, U[:,t],poly_order=poly_order, CPtype=CPtype, ref_H=ref_H)
        etime = time.time()
        print(f'Time elapsed for station {station_id}: {etime-stime}s')
        coeff_da = xr.DataArray(Coeff, dims=['time', 'coeff'], coords={'time': filtered_data.time, 'coeff': np.arange(poly_order+1)},name='Chebyshev_Coefficients')

        # Save the coefficients to a netCDF file
        output_dir = f'data/Profiler_Chebyshev_Coefficients'
        os.makedirs(output_dir, exist_ok=True)
        coeff_da.to_netcdf(f'{output_dir}/{station_id}.nc')
    except Exception as e:
        print(f'Error for station {station_id}: {e}')
        continue

