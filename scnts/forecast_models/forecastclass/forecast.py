import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from scnts.TS.ts_class import ts
from scnts.forecast_models.forecastclass.metrics import *
#from scnts.forecast_models.stat.statmodels import ETS
# To include: mean, new_ts, prediction, original_ts, mse, in_sample, residuals, 



def check_true_data(true_data, h, original_ts, full_ts):
    # If the user provided just the data
    if len(true_data) == h:
        if type(true_data) == list or type(true_data) == np.ndarray:
            # Concatenating
            data = np.concatenate([original_ts.data, true_data])
        elif type(true_data) == ts:
            data = np.concatenate([original_ts.data, true_data.data])
        else:
            raise AttributeError("Please provide the true data as a list or as a ts object.")
        # Getting the dates
        dates = original_ts.date_range
        # Converting to date_range
        if type(dates) != pd.core.indexes.period.PeriodIndex and type(dates) != pd.core.indexes.period.DatetimeIndex:
            dates = pd.DatetimeIndex(dates)
        # Moving forward the time steps
        new_dates = dates.shift(h, original_ts.frequency)
        new_dates = dates.union(new_dates)
        true_ts = ts(data, date_range = new_dates)
        
# If it is not none -> it has been reversed when forecasting
# If it is not reversed we cannot proceed
    elif full_ts != 'Not':
        # if it is the same length as with the reversed ts
        if len(true_data) == len(full_ts):
            if type(true_data) != ts:
                true_ts = ts(true_data)
            else:
                true_ts = true_data
        else:
            raise AttributeError("Length between the given ts and the forecasted object are missmatched")

    elif len(true_data) == h + len(original_ts):
        
        print("Warning! Forecasts are not reversed. Scales might be missmatched")
        data = np.concatenate([original_ts.data, true_data.data[-h:]]) # only the last h values
            # Getting the dates
        dates = original_ts.date_range
        data = original_ts.data
        # Converting to date_range
        if type(dates) != pd.core.indexes.period.PeriodIndex and type(dates) != pd.core.indexes.period.DatetimeIndex:
            dates = pd.DatetimeIndex(dates)           
            # Moving forward the time steps
        new_dates = dates.shift(h, original_ts.frequency)
        new_dates = dates.union(new_dates)
        true_ts = ts(data, date_range = new_dates)

    else:
        raise AttributeError("Incorrect format for the original data.")

    return true_ts



class forecast(object):

    def __init__(self, model, mean, residuals, in_sample_pred, in_sample_mse, original_ts = None,
                pi = 'Not', level = None, full_ts = 'Not', true_data = 'Not'): # true data might be given

        self.model = model
        self.mean = mean
        self.residuals = residuals
        self.in_sample_pred = in_sample_pred
        self.in_sample_mse = in_sample_mse
        self.original_ts = original_ts

        self.pi = pi
        self.full_ts = full_ts
        self.level = level
        

        self.h = len(mean)
        # Checking the format of the true data.
        if true_data != 'Not':
            self.true_ts = check_true_data(true_data, self.h, self.original_ts, self.full_ts)

   # Here adds the true function to evaluate
   # Also add it as aargument to the base models

   # True data can either be a ts object with the length equal to the new_ts
   # or an array equal to h
    def add_real(self, true_data):
        self.true_ts = check_true_data(true_data, self.h, self.original_ts, self.full_ts)  

    # in-sample weather to plot the in_sample predictions
    def plot(self, true_test = True, in_sample = True, PI = True):
        
        # Check the warnings here.
        if self.full_ts == 'Not':
            raise AttributeError("Can not plot without reversing forecasts due to differences in scales. Enable reverse_forecasts")

        if true_test:
            if type(self.true_ts) != ts:
                raise AttributeError("The true test values have not been properly provided. Please add them first.")



        # Getting the true data and the predictions
        fitted = self.original_ts.data
        x_fitted = self.original_ts.date_range
        # Getting the predicted
        predicted_outofsample = self.full_ts.data[-self.h:]
        y_fitted = self.full_ts.date_range[-self.h:]
        predicted_insample = self.in_sample_pred


        # Here check if the true values are enabled

        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(1, 1, 1)

        # for the vline
        x_line = self.full_ts.date_range[-self.h]

        # Prediction Interval
        if PI:
            if len(self.pi) == 1:
                raise AttributeError("Can not plot prediction intervals without calling them during forecasting. Enable PI estimation during forecasting")
            else:
                lower = self.pi[0]
                upper = self.pi[1]

        # The real train values
        plt.plot(x_fitted, fitted,  color = 'black', linewidth = 1.5, label = 'True Data')
        # If in_samples is enabled:
        if in_sample:
            plt.plot(x_fitted, predicted_insample,  color = 'blue', linewidth = 1.5)
        
        # A horizontal line to split the train and the test set
        plt.axvline(x_line, linestyle = '--', color = 'black', linewidth = 1.5)
        
        # The out-of-sample predictions
        plt.plot(y_fitted, predicted_outofsample, label = 'Predictions', linewidth = 2, color = 'blue')
        
        # For the test set
        if true_test:
            true_outofsample = self.true_ts.data[-self.h:]
            plt.plot(y_fitted, true_outofsample, linewidth = 2, color = 'black')


        # For The prediction interval
        if PI:
            plt.fill_between(y_fitted, y1 = upper, y2 = lower, alpha = 0.4, color = 'royalblue') 

        # Adding the details
        gray_scale = 0.93
        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        ax.grid(linestyle='-', color = 'w', linewidth = 2)
        ax.legend(loc = 2)
        periods = int(math.ceil(len(self.full_ts)/10))
        ax.xaxis.set_major_locator(plt.MaxNLocator(periods))

        plt.show()     


    # Return a dataframe with the Predictions, the PIs etc
    def summary(self):
        # Preparing the index
        horizon_index = np.arange(0, self.h )
        data = self.mean
        cols = ['Mean Predictions']
        # The mean predictions
        if len(self.pi) != 1:
            lower = self.pi[0]
            upper = self.pi[1]
            # merging
            data = np.vstack([data, lower])
            data = np.vstack([data, upper])
            cols.append('Lower Interval')
            cols.append('Upper Interval')
        if self.true_ts != 'Not':
            true = self.true_ts.data[-self.h:]
            data = np.vstack([data, true])
            cols.append('True Data')

        summarized = pd.DataFrame(data.T, index = horizon_index, columns = cols)
        return summarized

    # If only true data is given

    # Supported metrics so far: mse, rmse, mae, mape, smape, mase, rmsse
    # Will add some more
    # Other metrics can also be passed
    def evaluate(self, metrics = [mse, smape, mase, rmsse]):
        # Assert the true and the predicted values are properly given
        if type(self.true_ts) != ts:
            raise AttributeError("The true test values have not been properly provided. Please add them first.")
        if type(self.full_ts) != ts:
            raise AttributeError("Predicted data is not reversed. Please allow the forecasts to be reversed")       

        # Get the values
        pred = self.full_ts.data[-self.h:]
        true = self.true_ts.data[-self.h:]

        # Initialize
        value = []
        key = []
        
        values = [metric(true, pred) for metric in metrics]
        keys = [metric.__name__ for metric in metrics] # .__name__ returns the name of the function

        d = dict(zip(keys,values))
        # Converting to dataframe
        d = pd.DataFrame(d , index = [0]) #setting the index to deal with an error
        return d


