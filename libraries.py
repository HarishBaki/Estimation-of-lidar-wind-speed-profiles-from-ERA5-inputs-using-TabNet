
# === importing dependencies ===#
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import ast
import yaml
import time
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as R2
from scipy.interpolate import interp1d

# we would like to make sure that the reference height levels cover the profiler levels and NOW23 levels, including the 0 m level.
profilers_levels = np.array([10] + list(range(100, 501, 25)))
NOW23_levels = np.array([10] + list(range(20, 301, 20)) + [400, 500])
# combine the two arrays and remove duplicates
ref_H = np.unique(np.concatenate((np.array([0]),profilers_levels, NOW23_levels)))

# These variables are used while creating the Chebyshev coefficients
poly_order = 4
CPtype = 1

def normalize(H,ref_H=ref_H):
    '''
    Normalizes the height levels between -1 and 1
    ref_H: A vector of reference height levels, in our case CERRA levels
    H: A vector of height levels
    '''
    a = 2 / (np.max(ref_H) - np.min(ref_H))
    b = - (np.max(ref_H) + np.min(ref_H)) / (np.max(ref_H) - np.min(ref_H))
    Hn = a * ref_H + b

    Hn = np.interp(H, ref_H, Hn)
    return Hn

def Chebyshev_Basu(x, poly_order=poly_order, CPtype=CPtype):
    '''
    This function computes the Chebyshev polynomials, according to the equations in Mason, J. C., & Handscomb, D. C. (2002). Chebyshev polynomials. Chapman and Hall/CRC.
    x: input variable, between -1 and 1
    poly_order: order of polynomials
    CPtype: 1 or 2, according to the publication
    '''
    if x.ndim == 1:
        x = x[:, np.newaxis]

    CP = np.zeros((len(x), poly_order + 1))

    CP[:, 0] = 1  # T0(x) = 1
    if poly_order >= 1:
        if CPtype == 1:  # Chebyshev polynomial of first kind
            CP[:, 1] = x.flatten()  # T1(x) = x
        else:  # Chebyshev polynomial of second kind
            CP[:, 1] = 2 * x.flatten()  # T1(x) = 2x
        if poly_order >= 2:
            for n in range(2, poly_order + 1):
                CP[:, n] = 2 * x.flatten() * CP[:, n - 1] - CP[:, n - 2]
    return CP

def Chebyshev_Coeff(H, U,poly_order=poly_order,CPtype=CPtype,ref_H=ref_H):
    '''
    This function computes the Chebyshev coefficients through inverse transform of system of linear equations
    H: height levels, in their useual units
    U: wind speed at the height levels
    p: polynomial order
    CPtype: 1 or 2, according to the publication
    '''
    H = H.flatten()
    U = U.flatten()

    # Normalize H
    Hn = normalize(H, ref_H=ref_H)
    
    # Remove NaN values
    Indx = np.where(~np.isnan(U))[0]
    Ha = Hn[Indx]
    Ua = U[Indx]
    N = len(Ua)

    # Linearly extrapolate wind values at the boundaries
    #spline_left = interp1d(Ha, Ua, kind='linear', fill_value='extrapolate')
    #Uax = spline_left([-1])

    #spline_right = interp1d(Ha, Ua, kind='linear', fill_value='extrapolate')
    #Uay = spline_right([1])
    #Ua = np.concatenate([Uax, Ua, Uay])
    #Ha = np.concatenate([-1 + np.zeros(1), Ha, 1 + np.zeros(1)])       # these two seems are unnecessary, which bring adidtional offset due to extrapolation.
    
    # Predict the gap-filled and denoised profile
    PL = Chebyshev_Basu(Ha, poly_order=poly_order, CPtype=CPtype)
    # Compute the coefficients C
    Coeff = np.linalg.pinv(PL) @ Ua
    return Coeff

def WindProfile(Z,Coeff, poly_order=poly_order, CPtype=CPtype,ref_H=ref_H):
    '''
    This function computes the full level wind profile provided vertical levels and the Chebyshev coefficients
    Z: height levels, in their useual units
    Coeff: Chebysev coefficients
    '''
    # Normalize H
    Hn = normalize(Z, ref_H=ref_H)
    PL_full = Chebyshev_Basu(Hn, poly_order=poly_order, CPtype=CPtype)
    Mp = PL_full @ Coeff
    return Mp

def Basu_WindProfile(z, a,b,c):
    '''
    This function computes the full level wind profile provided vertical levels and the Basu coefficients
    Z: height levels, in their useual units
    a,b,c: Basu coefficients
    '''
    return 1 / (a + b * np.log(z) + c * (np.log(z))**2)

def Basu_Coeff(z,u):
    from scipy.optimize import curve_fit
    # Perform curve fitting
    popt, pcov = curve_fit(Basu_WindProfile, z, u, p0=[1, 1, 1])  # p0 are initial guesses for parameters a, b, and c
    return popt

def data_processing(input_file,ChSh_Coeff_file,input_times_freq,input_variables,target_variables, dates_range, station_id,val_arg=None,segregate_arg=None,rng_data=None):
    '''
    This function reads the nc files and converts them into numpy arrays in the required shape.
    input_file: input variables file (either ERA5 or CERRA)
    ChSh_Coeff_file: target variables file (Chebyshev coefficients file)
    input_time_freq: time frequency of the input variables, since the CERRA and ERA5 are not at the same time frequencey
    input_variables: names of the input variables
    target_variables: target Chebyshev coefficients 
    dates_range: range of the dates to read, can be training or testing
    locations: location indices of the data, out of 11
    var_arg: whether the function should return validation data also along with training, or only training, or testing
    '''
    inputs = xr.open_dataset(input_file)
    ChSh_Coeff = xr.open_dataset(ChSh_Coeff_file)
    '''
    There are two conditioning we have to consider:
    1. We have to segregate the data based on the outliers in the Chebyshev coefficients, thus take the time indices of the Chebyshev coefficients as base.
    2. We have to eliminate the NaN values in the input ERA5 variables, thus take the time indices of ERA5 inputs.
    To balance, we have to take the intersection of the two time indices.
    '''
    X = inputs[input_variables].sel(location=station_id).to_array() # X is a 2D array, with the first dimension as the input variables.
    Y = ChSh_Coeff.sel(coeff=target_variables).to_array()   # Remember, Y is a 3D array, with the first dimension as the Chebyshev coefficients, then time, and coeff.
    Y = Y.isel(variable=0).drop('variable') # drop the variable dimension, since it is not needed.
    Input_notmissing_mask = (X.sel(valid_time=slice(*dates_range,input_times_freq))).notnull().all(dim='variable')
    input_time_coord = X.sel(valid_time=slice(*dates_range,input_times_freq)).valid_time.where(Input_notmissing_mask,drop=True)
    if segregate_arg:
        target_time_coord = Y.sel(time=slice(*dates_range,input_times_freq)).where(ChSh_Coeff.outlier==1,drop=True).coords['time']
        print('Segregated times:',target_time_coord.size)
    else:
        target_time_coord = Y.sel(time=slice(*dates_range,input_times_freq)).coords['time']
        print('All times:',target_time_coord.size)
    # Now intersecting them both. I cannot do np.intersect1d, since it converts the data into numpy array, and I need the xarray data.
    time_coord = target_time_coord.reindex(time=input_time_coord.valid_time.values).dropna(dim='time')
    print('Intersected times:', time_coord.size)
    if val_arg:
        #=== Extracting training and validation indices ===# 
        years = time_coord.dt.year
        months = time_coord.dt.month
        validation_times = np.zeros(len(time_coord), dtype=bool)
        years = time_coord.dt.year
        months = time_coord.dt.month
        validation_times = np.zeros(len(time_coord), dtype=bool)
        for year in np.unique(years):
            for month in range(1, 13):
                # check if you have enough data in the month
                month_indices = np.where((years == year) & (months == month))[0]
                '''
                One problem we have to deal with is not enough data points in a month.
                Not always, the number of samples in a month is enough to take 6 days of data for validation.
                Thus, lets consider 20% of the month data for validation.
                '''
                validation_window = int(0.2*len(month_indices))
                try:
                    start_index = rng_data.choice(len(month_indices) - validation_window - 1)
                    #print('start_index:',start_index)
                    validation_indices = month_indices[start_index:start_index + validation_window]
                    validation_times[validation_indices] = True
                except:
                    pass
        train_time_coord = time_coord.sel(time=~validation_times,drop=True)
        valid_time_coord = time_coord.sel(time=validation_times,drop=True)
        # --- training ---#
        X_train = X.sel(valid_time=train_time_coord.values).values.T
        Y_train = Y.sel(time=train_time_coord.values).values

        # --- vlaidation ---#
        X_valid = X.sel(valid_time=valid_time_coord.values).values.T
        Y_valid = Y.sel(time=valid_time_coord.values).values       
        
        return X_train, Y_train, X_valid, Y_valid, train_time_coord, valid_time_coord

    else:
        # --- testing ---#
        X_test = X.sel(valid_time=time_coord.values).values.T
        Y_test = Y.sel(time=time_coord.values).values
        return X_test, Y_test, time_coord

def L1_loss(y_pred, y_true):
    """
    Custom L1 loss function for a 5-coefficient target vector.
    """
    return torch.mean(torch.abs(y_pred - y_true))

def MSE_loss(y_pred, y_true):
    """
    Custom MSE loss function for a 5-coefficient target vector.
    """
    return torch.mean((y_pred - y_true) ** 2)

def weighted_MSE_loss(y_pred, y_true):
    """
    Custom weighted MSE loss function for a 5-coefficient target vector.
    Emphasizes errors on the last three coefficients.
    """
    # Define the weights, with more weight on the last three coefficients
    weights = torch.tensor([0.5, 1.0, 5.0, 10.0, 10.0], device=y_pred.device)

    # Compute the squared differences
    squared_diff = (y_pred - y_true) ** 2

    # Apply the weights to the squared differences
    weighted_squared_diff = squared_diff * weights

    # Compute the mean of the weighted squared differences
    return torch.mean(weighted_squared_diff)

def focal_MSE_loss(y_pred, y_true):
    """
    Custom focal MSE loss for weighted regression.
    
    Parameters:
    - y_pred: Predicted values (shape: [batch_size, 5])
    - y_true: Ground truth values (shape: [batch_size, 5])
    - weights: Tensor of weights to emphasize specific coefficients (shape: [5])
    - gamma: Focusing parameter to increase weight on larger errors
    
    Returns:
    - Loss value (scalar)
    """
    # Define the weights, with more weight on the last three coefficients
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=y_pred.device)

    # Compute the squared differences (MSE for each coefficient)
    squared_diff = (y_pred - y_true) ** 2
    
    # Apply the weights to emphasize specific coefficients
    weighted_diff = squared_diff * weights
    
    # Apply the focal modulating factor
    focal_weight = (1 - torch.exp(-weighted_diff)) ** 0.5
    focal_loss = focal_weight * weighted_diff
    
    # Return the mean loss across the batch and coefficients
    return focal_loss.mean()

def profiler_loss(Y_pred,Y_true):
    '''
    This function computes the full level wind profile provided vertical levels and the Chebyshev coefficients
    Z: height levels, in their useual units
    Coeff: Chebysev coefficients
    '''
    # Normalize H
    Z = np.linspace(10,500+1,20)
    Hn = normalize(Z, ref_H=ref_H)
    PL_full = Chebyshev_Basu(Hn, poly_order=poly_order, CPtype=CPtype)
    PL_full = torch.tensor(PL_full, dtype=torch.float32,device=Y_pred.device)
    Profile_pred = (PL_full @ Y_pred.T).T
    Profile_true = (PL_full @ Y_true.T).T

    return torch.sqrt(torch.mean((Profile_pred - Profile_true)**2))

def Kho_loss(y_pred, y_true):
    """
    Custom Kho loss function for a 5-coefficient target vector.
    Designed based on https://doi.org/10.1007/s00703-020-00736-3.
    """
    numerator = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    denominator = torch.sqrt(torch.var(y_pred)+torch.var(y_true))

    return numerator/denominator 

def Kho_loss_on_profile(Y_pred,Y_true):
    '''
    This function computes the full level wind profile provided vertical levels and the Chebyshev coefficients
    Z: height levels, in their useual units
    Coeff: Chebysev coefficients
    '''
    # Normalize H
    Z = np.linspace(10,500+1,20)
    Hn = normalize(Z, ref_H=ref_H)
    PL_full = Chebyshev_Basu(Hn, poly_order=poly_order, CPtype=CPtype)
    PL_full = torch.tensor(PL_full, dtype=torch.float32,device=Y_pred.device)
    Profile_pred = (PL_full @ Y_pred.T).T
    Profile_true = (PL_full @ Y_true.T).T

    numerator = torch.sqrt(torch.mean((Profile_pred - Profile_true) ** 2))
    denominator = torch.sqrt(torch.var(Profile_true)+torch.var(Profile_true))

    return numerator/denominator 