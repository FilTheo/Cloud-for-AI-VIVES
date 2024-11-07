from numpy.core.numeric import full
from scnts.TS.ts_class import ts
from scnts.forecast_models.forecastclass.forecast import forecast
#from scnts.forecast_models.ml.forecastingutils import 
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
import numpy as np

from scnts.forecast_models.ml.forecastingutils import autoregressive_forecasts, get_errors, reverse_forecasts, direct_quant


# The base class for ml models

class MLBaseModel(object):
    
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    
# The Fit function
# Original ts should be provided for reversing predictions
    def fit(self, lagged_ts, original_ts, true_ts = 'Not'): # user can provide the test set as well.. for evaluation 

        self.lagged_ts = lagged_ts
        self.original_ts = original_ts
        self.true_ts = true_ts
        
        # Splitting target/features
        x_train = lagged_ts.loc[:, lagged_ts.columns != 'y'].values
        y_train = lagged_ts['y'].values

        # Fitting
        self.model = self.picked_model.fit(x_train, y_train)

    def forecast(self, h, reverse_forecast = True, PI = True, level = 95, pi_method = 'empirical-direct'):
        # Making a copy of the original ts
        fitted = ts(self.original_ts.ts.copy())
        self.fitted = fitted
        
        # Generating predictions
        predictions = autoregressive_forecasts(self.model, self.lagged_ts, self.fitted, h)
        if reverse_forecast:
            self.new_ts, mean = reverse_forecasts(predictions, self.fitted, self.lagged_ts, h)
            self.mean = mean.data
            self.not_reversed_mean = predictions
        else:
            self.mean = predictions
            self.new_ts = 'Not'
        
        # Getting residuals
        errors = get_errors(self.model, h, self.fitted, self.lagged_ts)
        self.residuals = errors[:,1]
        # Getting the in-sample forecasts
        # Attention: first observations from the in-sample will be the same with the fitted data.
        in_sample_pred = self.fitted.data[-len(self.residuals):] - self.residuals
        
        # Calculating in-sample mse 
        self.mse = mse(self.fitted.data[-len(self.residuals):] , in_sample_pred )
        
        # Concatenating with the true
        self.insample_pred = np.concatenate([self.fitted.data[:-len(self.residuals)], in_sample_pred])


        if PI:
            if pi_method == 'empirical-direct':
                quantiles = direct_quant(errors, level)
            # Currently supporting only the direct.
            # Will introduce other methods soon
            else:
                raise AttributeError('Currently supporting only the direct empirical method. Soon more will be added')
            pis = [self.mean[i] + quantiles[i] for i in range(len(quantiles))]
            self.pi = np.array(pis).T
            self.level = level
        else:
            self.pi = 'Not' #for the forecast object
            self.level = None

        #Calling the forecast object to return
        fc = forecast(
            model = self.model,
            mean = self.mean,
            residuals = self.residuals,
            in_sample_pred = self.insample_pred,
            in_sample_mse = self.mse,
            original_ts = self.fitted,
            pi = self.pi,
            level = self.level,
            full_ts = self.new_ts,
            true_data = self.true_ts )
        return fc
