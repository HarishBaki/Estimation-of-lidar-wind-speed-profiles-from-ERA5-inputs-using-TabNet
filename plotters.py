import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle


from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as R2

from libraries import *

def EMD(df1,df2):
    bins = np.arange(0,25.1,0.1)
    hist1 = np.histogram(df1,bins,density=True)[0]
    hist2 = np.histogram(df2,bins,density=True)[0]
    #return stats.wasserstein_distance(bins[:-1],bins[:-1],u_weights=hist1,v_weights=hist2)
    return stats.wasserstein_distance(df1,df2)

def hexbin_plotter(fig,gs,Y,pred,title,text_arg=None,xlabel=None,ylabel=None):
    '''
    Plots hebxin between true and predictions of Y
    fig: figure handle
    gs: grid spect handle
    Y: target (train or test or true) 
    pred: prediction from a model
    title: title of the figure
    text_arg: whether to add text with in the plot or not
    xlabel_arg: some cases, the xlabel is not needed, this specifies that
    ylabel_arg: some cases, the ylabel is not needed, this specifies that 
    '''
    errMAE    = mae(Y,pred)
    errRMSE   = np.sqrt(mse(Y,pred))
    errMAPE   = mape(Y,pred)
    errR2     = R2(Y,pred)
    #errEMD      = EMD(Y,pred)

    ax_hexbin = fig.add_subplot(gs)
    hb = ax_hexbin.hexbin(np.squeeze(Y), np.squeeze(pred), gridsize=100, bins='log', cmap='inferno')
    if text_arg:
        ax_hexbin.text(0.05, 0.93, f'MAE: {errMAE:.2f} \n$R^2$: {errR2:.2f}\nRMSE: {errRMSE:.2f} \nMAPE: {errMAPE:.2f}',
                      transform=ax_hexbin.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if xlabel:
        ax_hexbin.set_xlabel(xlabel)
    if ylabel:
        ax_hexbin.set_ylabel(ylabel)
    ax_hexbin.set_title(f'{title}')

    min_value = Y.min()
    max_value = Y.max()
    ax_hexbin.set_xlim(min_value, max_value)
    ax_hexbin.set_ylim(min_value, max_value)
    ax_hexbin.plot([min_value, max_value], [min_value, max_value], 'k--')