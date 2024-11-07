from scnts.forecast_models.forecastclass.metrics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scnts.forecast_models.stat.stat_base_model import StatisticalBaseModel






# A function to compare up to 5 forecasts
# accepts two or more forecast objects

# forecasts is an array

class compare_forecasts(object):

    def __init__(self, forecasts, metrics = [mse, smape, mase, rmsse], true_set = 'Not'):
        
        # For more than 5 forecasts raise error
        if len(forecasts) > 5:
            raise AttributeError("Only supporting comparissons between 5 or less models")
        elif len(forecasts) == 1:
            raise AttributeError("Can not compare a model with itself. Add more models or use another evaluation method")

        self.forecasts = forecasts
        self.metrics = metrics
        # getting the true data from the first forecast object
        self.fitted = self.forecasts[0].original_ts
        self.h = self.forecasts[0].h
        if true_set == 'Not':
            try:
                # Taking the true data from the first forecast object
                self.true_set = self.forecasts[0].true_ts
            except:
                raise AttributeError("True set not provided for evaluation")


    def plot(self, in_sample = False):
        
         
        fitted = self.fitted.data
        x_fitted = self.fitted.date_range

        true = self.true_set.data[-self.h:]
        y_fitted = self.true_set.date_range[-self.h:]

        total_length = len(x_fitted) + len(y_fitted)
        # Prepare and plot the real values
        fig = plt.figure(figsize = (16,8))
        ax = fig.add_subplot(1, 1, 1)
        x_line = y_fitted[0]

        plt.plot(x_fitted, fitted, color = 'black', linewidth = 1.5, label = 'True Data')
        plt.plot(y_fitted, true, color = 'black', linewidth = 1.5)
        plt.axvline(x_line, linestyle = '--', color = 'black', linewidth = 1.5)

        
        # colors = [] -> pick the colors
        # for fc,col in zip(self.forecasts,colors)
        # Plots the predictions
        for fc in self.forecasts:
            predicted = fc.mean
            model = fc.__class__.__name__ if isinstance(fc,StatisticalBaseModel) else fc.model.__class__.__name__ 
            plt.plot(y_fitted, predicted,  linewidth = 1.5, label = model) #also add the color 
            if in_sample:
                insample = fc.in_sample_pred
                plt.plot(x_fitted, insample,  linewidth = 1.5) # same color here

        
        gray_scale = 0.93
        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        ax.grid(linestyle='-', color = 'w', linewidth = 2)
        ax.legend(loc = 2)
        periods = int(math.ceil(total_length/10))
        ax.xaxis.set_major_locator(plt.MaxNLocator(periods))

        plt.show()

    def summary(self):

        # Initializing
        s = (len(self.forecasts), len(self.metrics))
        score_mat = np.zeros(s)
        cols = [metric.__name__ for metric in self.metrics]
        models = [fc.__class__.__name__ if isinstance(fc,StatisticalBaseModel) else fc.model.__class__.__name__  for fc in self.forecasts ]
        true = self.true_set.data[-self.h:]

        for i in range(len(self.forecasts)):
            pred = self.forecasts[i].mean
            score_mat[i] = [metric(true, pred) for metric in self.metrics]

        d = pd.DataFrame(score_mat, columns = cols, index = models)
        return d