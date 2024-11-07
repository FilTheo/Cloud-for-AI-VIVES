import numpy as np
import re

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from scnts.TS.ts_utils import get_numeric_frequency
from scnts.local_preprocess.seasonal_features import *
from scnts.forecast_models.ml.reverseforecasts import check_transformations, seasonal_stationary, stationary, stationary_reverse
from scnts.TS.ts_class import ts



def PolynomialRegression(degree = 2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# Builds a single x_test row
def get_test_row(lagged_ts, original_ts, h, i, exog_feat_bool = False):
    # will generate the lags for the next value to forecast 
    # getting the lags 

    cols = lagged_ts.columns.values
    # Removing the true value
    cols = cols[cols != 'y']
    # Getting the lag cols
    lags_cols = [col for col in cols if 'lag' in str(col)]

    # getting how many steps back to look to get the values
    # Simply extracting each number from the lag cols
    steps = [int(re.findall(r'\d+', col)[0]) for col in lags_cols ]
    # reversing indices to get to start from the end
    reverse_steps = [len(lagged_ts) - step for step in steps]
    # Getting the lagged values
    y_values = lagged_ts['y'].values
    # Taking the values at each lag position
    lags = np.take(y_values, reverse_steps)

    # Checking for exogenus features (fourier terms or dummies)
    # Also checking if exogenus features are True (this is used on prediction interval estimation)
    if (len(cols) != len(lags_cols)) and exog_feat_bool == False:
        # getting the exogenus features
        extra_cols = list(set(cols) - set(lags_cols))
        # Checking if its fourier or dummies
        # cos-0 will always be there
        if 'COS-0' in extra_cols:
            # will generate the features

            # Getting the order of the fourier terms 
            order = int(len(extra_cols)/2)

            # getting the dates, the values and the frequency
            data = original_ts.data
            dates = original_ts.date_range
            order = 1
            h = 12
            numeric_freq = int(get_numeric_frequency(original_ts))
            # generating fourier
            fourier_series = FourierFeaturizer(numeric_freq, order)
            # Transforming
            dummy, exog = fourier_series.fit_transform(data)
            # Getting the new terms
            a , exog_to_add = fourier_series.transform(y = None, n_periods = h)
            exog_to_add = exog_to_add.values[i,:].reshape(-1)
            # Appending to the lags
            lags = np.concatenate([lags, exog_to_add], axis = 0)
            

        #if we have dummies
        #else 
        else:
            # get the last row to determine the next position of the dummy
            last_row = lagged_ts[extra_cols].values[-1]
            # Finding the location of the dummy 
            idx = np.argwhere(last_row == 1).reshape(1)[0]
            # Initialize new row 
            new_row = np.zeros(len(last_row))

            # Adding the dummy in the next value from the idx
            # Using mod to cover the case where the dummy is in the last position so it moves to the first 
            new_idx = (idx + 1)%len(last_row)
            new_row[new_idx] = 1
            # Appending to lags
            lags = np.concatenate([lags, new_row], axis = 0)
    # If exogenus features are passed as True then I use them for PI estimation
    # The original_ts parameter is used to get the exogenus features
    elif (len(cols) != len(lags_cols)) and exog_feat_bool == True:
        # Extracting the exogenus features
        exog_cols = original_ts.columns.values
        exog_cols = exog_cols[exog_cols != 'y']
        # Removing the 'y' column
        extra_cols = list(set(exog_cols) - set(lags_cols))
        # Making the new exog_features with only the kept columns and taking the values
        exog_to_add = original_ts[extra_cols].values[i,:]
        # Appending
        lags = np.concatenate([lags, exog_to_add], axis = 0)
        
    return lags


def autoregressive_forecasts(model, lagged_ts, original_ts, h, exog_feat_bool = False):
    # Initialize
    forecasts = np.zeros(h)
    
    for i in range(h):
        # create a new x_train row with the get_test_row
        test_row = get_test_row(lagged_ts , original_ts, h, i, exog_feat_bool)
        # Reshaping for predictions
        to_predict = test_row.reshape(1,-1)
        #Making the prediction
        prediction = model.predict(to_predict)

        # Appending to predictions
        forecasts[i] = prediction
        # Appending to the new entry
        new_row = np.insert(test_row, 0, prediction)
        # Preparing to add into the array 

        # Adding the new entry to the ts
        #lagged
        new_row = pd.DataFrame(new_row.reshape(1,-1), columns = lagged_ts.columns)
        lagged_ts = lagged_ts.append(new_row, ignore_index = True)
    return forecasts


def reverse_forecasts(forecasts, original_ts, lagged_ts, h ):
    # Taking the transformations
    normalizer, stationarity, log, total_tfs =  check_transformations(original_ts, lagged_ts)
    # Preparing the array to reverse
    # if we have normalized, reversing the normalization
    if normalizer != None:
        forecasts = normalizer.inverse_transform(forecasts.reshape(-1,1)).reshape(-1)
    
    stat = seasonal_stationary(original_ts, print_results = False)
    stat = stationary(stat, print_results = False)
    ts_for_reverse = np.concatenate([stat.data,forecasts])
    # Reversing the stationarity
    reversed_series = stationary_reverse(original_ts, ts_for_reverse, total_tfs, log )

    reversed_forecasts = ts(reversed_series.ts.iloc[-h:,:])
    return reversed_series, reversed_forecasts


# For prediction intervals

def get_errors(model, h, original_ts, lagged):

    # For reversing
    normalizer, stationarity, log, total_tfs =  check_transformations(original_ts, lagged)
    # Initiating
    errors_mat = np.zeros((len(lagged) - (2*h),h))

    # - 2*h to stop on the last values
    #  This is due to cross validation reaching the limit 
    for i in range(len(lagged) - (2*h) ):
        # Taking just the data
        lagged_data = list(lagged['y'].values)

        # Taking the x_test for predicting
        train_data = lagged.iloc[i:(h + 1) + i,:].copy()
        # This is to get the exog features
        exog_feat = lagged.iloc[(h + 1) + i:(2*h + 1) + i ,:].copy()
        # Getting the in-sample forecasts
        in_sample_fc = autoregressive_forecasts(model, train_data, exog_feat, h, exog_feat_bool = True )

        # Replacing the in-sample predictions into the list of values
        idx = np.arange((h + 1) + i,(2*h + 1) + i )

        lagged_data[(h + 1) + i:(2*h + 1) + i] = in_sample_fc

        # Reversing the normalization
        if normalizer != None:
            lagged_data = normalizer.inverse_transform(np.array(lagged_data).reshape(-1,1)).reshape(-1)
        # Reversing the stationarity 
        
        stat = seasonal_stationary(original_ts, print_results = False)
        stat = stationary(stat, print_results = False)

        # finding how many values are missing from the beginning of the lagged_data
        # data are missing due to the Nan values on the create_lags functions or due to the stationarity transformations
        to_add = len(stat.data) - len(lagged_data)

        #Adding them and reversing
        lagged_data = np.concatenate([stat.data[:to_add],lagged_data])
        reversed_series = stationary_reverse(original_ts, lagged_data, total_tfs, log )

        # removing some values again due to lags and stationarity from both the reversed and the original
        to_add = len(reversed_series.data)-len(lagged)
        predicted_data = reversed_series.data[to_add:]
        true_data = original_ts.data[to_add:]

        # taking the values from this itteration
        predicted_data = predicted_data[(h + 1) + i:(2*h + 1) + i]
        true_data = true_data[(h + 1) + i:(2*h + 1) + i]

        # Taking the errors and appending
        errors = true_data - predicted_data

        errors_mat[i] = errors
    
    return errors_mat

# Direct extraction of upper and lower quantile
def direct_quant(errors, level):
    
    if level > 1:
        level = level/100

    h = errors.shape[1]
    #Initializing quantiles:
    quantiles = np.zeros((h ,2))
    
    #For every forecasting horizon
    for i in range(h):

        #getting the upper and lower quantiles
        upper = np.quantile(errors[:,i], level)
        lower = np.quantile(errors[:,i], 1 - level)
        
        # Appending
        quantiles[i,0] =  lower
        #the upper interval
        quantiles[i,1] = upper
    return quantiles



