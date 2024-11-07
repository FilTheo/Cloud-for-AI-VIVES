# Includes the metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# a small epsilon to avoid devisions with zero
e = 1e-10


def simple_error(actual, predicted):
    # Simple difference
    return actual - predicted

def percentage_error(actual, predicted):
    # % Error
    return simple_error(actual, predicted)/ (actual + e) #The small e asserts that division is not with 0

def naive_forecasts(actual, lag = 1):
    # Just repeats previous samples
    return actual[:-lag]


def mse(actual, predicted):
    # The mse
    return mean_squared_error(y_true = actual, y_pred = predicted)


def rmse(actual, predicted):
    # for rmse just turn squared to false
    return mean_squared_error(y_true = actual, y_pred = predicted, squared = False)

def mae(actual, predicted):
    return mean_absolute_error(y_true = actual, y_pred = predicted)


def mape(actual, predicted):
    # !!!!!!!Carefull!!!!!!
    # MAPE is biased as it is not symmetric
    # MAPE is not defined for actual = 0
    error = np.abs(percentage_error(actual,predicted))
    return np.mean(error)


def smape(actual, predicted):
    # Symmetric mape
    error = 2.0 * np.abs(actual - predicted)/((np.abs(actual) + np.abs(predicted)) + e)
    return np.mean(error)

def mase(actual, predicted, lag = 1):
    # can be configured for snaive with lag = freq
    # computed with naive forecasting for lag = 1
    # Can not be computed for predictions with just 1 sample
    
    num = mae(actual, predicted)
    denom = mae(actual[lag:], naive_forecasts(actual, lag))
    return num/denom

def rmsse(actual, predicted, lag = 1):
    # root mean squared scaled error
    num = simple_error(actual, predicted)
    denom = mae(actual[lag:], naive_forecasts(actual, lag))
    error = np.abs(num/denom)
    return np.sqrt(np.mean(np.square(error)))




