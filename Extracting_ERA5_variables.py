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

    profiler_stations = pd.read_csv('data/profiler_locations.csv',usecols=[0,3,4])

    # --- Extracting surface level variables ---#
    par_names = ['u10', 'v10', 'u100', 'v100','zust','i10fg',
                't2m','skt','stl1','d2m','msl','blh','cbh',
                'ishf','ie','tcc','lcc','cape','cin','bld']
    
    for par_name in (par_names):
        # create a folder with name par inside data_dir
        par_dir = f'data/ERA5_variables/{par_name}'
        os.makedirs(par_dir, exist_ok=True)
        # --- extract data at each year ---#
        for year in np.arange(2018,2019+1):
            datasets = []
            for loc in range(len(profiler_stations)):
                ds = era5_sfc_hourly(year, par_name, 
                                    profiler_stations['lat [degrees]'][loc], 
                                    profiler_stations['lon [degrees]'][loc],
                                    profiler_stations['stid'][loc])
                datasets.append(ds.compute())
                del(ds)
            # Concatenate datasets along a new dimension ('location')
            combined_dataset = xr.concat(datasets, dim='location')
            combined_dataset['year'] = year
            file_path = f'{par_dir}/{year}.nc'
            if os.path.exists(file_path):
                os.remove(file_path)
            combined_dataset.to_netcdf(file_path)
            del(combined_dataset)
            print(par_name,year)
        # --- combining all years data ---#
        ds = xr.open_mfdataset(f'{par_dir}/*.nc', 
                                    parallel=True)
        file_path = f'data/ERA5_variables/{par_name}.nc'
        if os.path.exists(file_path):
            os.remove(file_path)
        ds.to_netcdf(file_path)
        print(par_name)

    # --- Extracting pressure level variables ---#
    lvls = [1000,975,950]
    pars = ['u','v','t']
    pars = ['t']
    for par in pars:
        for level in lvls:
            # create a folder with name par and level inside data_dir
            par_dir = f'data/ERA5_variables/{par}_{level}'
            os.makedirs(par_dir, exist_ok=True)
            # --- extract data at each year ---#
            for year in np.arange(2018,2019+1):
                datasets = []
                for loc in range(len(profiler_stations)):
                    ds = era5_pres_hourly(year, par,level, 
                                        profiler_stations['lat [degrees]'][loc], 
                                    profiler_stations['lon [degrees]'][loc],
                                    profiler_stations['stid'][loc])
                    datasets.append(ds.compute())
                    del(ds)
                # Concatenate datasets along a new dimension ('location')
                combined_dataset = xr.concat(datasets, dim='location')
                combined_dataset['year'] = year
                file_path = f'{par_dir}/{year}.nc'
                if os.path.exists(file_path):
                    os.remove(file_path)
                combined_dataset.to_netcdf(file_path)
                del(combined_dataset)
                print(par,level,year)
            # --- combining all years data ---#
            ds = xr.open_mfdataset(f'{par_dir}/*.nc',
                                        parallel=True)
            file_path = f'data/ERA5_variables/{par}_{level}.nc'
            if os.path.exists(file_path):
                os.remove(file_path)
            ds.to_netcdf(file_path)
            print(par)