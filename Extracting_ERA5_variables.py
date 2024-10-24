import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, re, time, math, calendar

import dask
import dask.distributed as dd

# Function to find the closest index in a 1D array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def era5_pres_hourly(year, par, level, target_lat,target_lon,location):
    files = sorted(glob.glob(f'/data/harish/Estimation-of-lidar-wind-speed-profiles-from-ERA5-inputs-using-TabNet/data/{year}/PRES*'))
    hr_data = xr.open_mfdataset(files,combine='nested', concat_dim='valid_time')
    # remiving unnecessary multiple dimensions names
    hr_data = hr_data.drop_vars(['number','expver'])

    hr_cor_data = hr_data[par].sel(pressure_level=level).sel(latitude=target_lat, 
                      longitude=target_lon, method='nearest').drop_vars('pressure_level')
    hr_cor_data['location'] = location
    hr_cor_data = hr_cor_data.rename(f'{par}_{level}')
    return hr_cor_data

def era5_sfc_hourly(year,par,target_lat,target_lon,location):
    files = sorted(glob.glob(f'/data/harish/Estimation-of-lidar-wind-speed-profiles-from-ERA5-inputs-using-TabNet/data/{year}/SFC*'))
    hr_data = xr.open_mfdataset(files,combine='nested', concat_dim='valid_time')
    # remiving unnecessary multiple dimensions names
    hr_data = hr_data.drop_vars(['number','expver'])
    hr_cor_data = hr_data[par].sel(latitude=target_lat, 
                      longitude=target_lon, method='nearest')
    hr_cor_data['location'] = location
    return hr_cor_data

if __name__ == '__main__':
    # Get inputs from sys.argv
    if len(sys.argv) == 1:
        sys.argv = ['', 2019, 'u', 975]    # for debugging
    input_year = int(sys.argv[1])
    input_par = sys.argv[2]
    input_level = int(sys.argv[3]) if len(sys.argv) > 3 else None

    profiler_stations = pd.read_csv('data/profiler_locations.csv', usecols=[0, 3, 4])

    # Extract surface variables
    if input_level is None:
        par_dir = f'data/ERA5_variables/{input_par}'
        os.makedirs(par_dir, exist_ok=True)
        datasets = []
        for loc in range(len(profiler_stations)):
            ds = era5_sfc_hourly(input_year, input_par, 
                                 profiler_stations['lat [degrees]'][loc], 
                                 profiler_stations['lon [degrees]'][loc],
                                 profiler_stations['stid'][loc])
            datasets.append(ds.compute())
            del(ds)
        combined_dataset = xr.concat(datasets, dim='location')
        combined_dataset['year'] = input_year
        file_path = f'{par_dir}/{input_year}.nc'
        if os.path.exists(file_path):
            os.remove(file_path)
        combined_dataset.to_netcdf(file_path)
        del(combined_dataset)
        print(input_par, input_year)
    
    # Extract pressure level variables
    else:
        par_dir = f'data/ERA5_variables/{input_par}_{input_level}'
        os.makedirs(par_dir, exist_ok=True)
        datasets = []
        for loc in range(len(profiler_stations)):
            ds = era5_pres_hourly(input_year, input_par, input_level, 
                                  profiler_stations['lat [degrees]'][loc], 
                                  profiler_stations['lon [degrees]'][loc],
                                  profiler_stations['stid'][loc])
            datasets.append(ds.compute())
            del(ds)
        combined_dataset = xr.concat(datasets, dim='location')
        combined_dataset['year'] = input_year
        file_path = f'{par_dir}/{input_year}.nc'
        if os.path.exists(file_path):
            os.remove(file_path)
        combined_dataset.to_netcdf(file_path)
        del(combined_dataset)
        print(input_par, input_level, input_year)