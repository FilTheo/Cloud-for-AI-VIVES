# The base class fors statistical models

from scnts.forecast_models.stat.rconverters import initiate_r
from scnts.forecast_models.forecastclass.forecast import forecast
import numpy as np
import pandas as pd

import rpy2.robjects.packages as rpackages
from rpy2 import rinterface, robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri

from scnts.TS.ts_class import ts
import matplotlib.pyplot as plt
import math

initiate_r()
forecast_r = importr('forecast')


def statistical_forecast(model, h, PI = True, level = 95, simulate = False, bootstrap = False):
    """ The forecast function for statistical models.

    Args:
        fitted_model (StatisticalBaseModel): A statistical model to use for forecasting
        h (int): The forecasting horizon
        PI (bool, optional): Whether to produce a PI or not. Defaults to True.
        level (int, optional): The level for the PI. Defaults to 95.
        simulate (bool, optional): Whether to use the simulation method for estimating prediction intervals. Defaults to False.
        bootstrap (bool, optional): Whether to use the bootstrap method for estimating prediction intervals. Defaults to False.

    Returns:
        list: A list including the mean forecast, the residuals, the in-sample mse and the lower and upper quantile
    """
    
    # The forecast object
    #model = fitted_model() # convething the method to an object
    fc =  forecast_r.forecast(model, h = h, PI = PI, level = level, simulate = simulate, bootstrap = bootstrap)
    # Point forecasts
    point_forecast = np.array(fc.rx('mean'))[0]
    # Getting the residuals and the in sample mse
    residuals_vals = np.array(fc.rx('residuals'))[0]
    mse_vals = np.mean([res**2 for res in residuals_vals])
    fitted_vals = np.array(model.rx('fitted')[0])
    
    # If we dont estimate the PI
    if not PI:
        return point_forecast, residuals_vals, mse_vals
    # We return the quantiles as well
    else:
        #Extracting the quantiles
        # The rx2 is equivelant to [[]] -> double parenthesin R
        upper_quantile = np.array(fc.rx2('upper')).reshape(-1)
        lower_quantile = np.array(fc.rx2('lower')).reshape(-1)
        return point_forecast, fitted_vals, residuals_vals, mse_vals, lower_quantile, upper_quantile


# The base model for the statistical methods
# Inherits from the base model

# Another option is to skip the model.py in general and just make the plot as a function callable by both ml and statistical


class StatisticalBaseModel(object):

    def __init__(self,*args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    
    # The abstract fit function.
    # Will be overriden by every model
    def fit(self):
        pass
    
    def forecast(self, h, PI = True, level = 95, simulate = False, bootstrap = False):
        
        self.h = h
        self.PI = PI
        self.level = level
        self.simulate = simulate
        self.bootstrap = bootstrap
        # Producing the forecasts
        statistical_predictions = statistical_forecast(self.model, self.h, 
                                                        PI = self.PI, level = self.level,
                                                        simulate = self.simulate, bootstrap = self.bootstrap)
        
        self.mean = statistical_predictions[0]
        self.insample_pred = statistical_predictions[1]
        self.residuals = statistical_predictions[2]
        self.mse = statistical_predictions[3]
        if PI:
            self.pi = (statistical_predictions[4], statistical_predictions[5])
        else:
            self.pi = 'Not'
            self.level = None 

        # train data + predictions
        new_data = np.concatenate([self.original_ts.data, self.mean])
        dates = self.original_ts.date_range
        # converting if not in the right format
        if type(dates) != pd.core.indexes.period.PeriodIndex and type(dates) != pd.core.indexes.period.DatetimeIndex:
                dates = pd.DatetimeIndex(dates)
        
        new_dates = dates.shift(self.h, self.original_ts.frequency)
        new_dates = dates.union(new_dates)
        self.new_ts = ts(new_data, date_range = new_dates)

        fc = forecast(
            model = self.model,
            mean = self.mean,
            residuals = self.residuals,
            in_sample_pred = self.insample_pred,
            in_sample_mse = self.mse,
            original_ts = self.original_ts,
            pi = self.pi,
            level = self.level,
            full_ts = self.new_ts,
            true_data = self.true_ts )
        return fc
    
    
    
    