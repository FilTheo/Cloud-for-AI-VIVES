# Creates and selects lags

from scnts.TS.ts_utils import get_numeric_frequency
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
from statsmodels.tsa.stattools import pacf



def create_lags(time_series, lags = 'default', remove_zeros = True):
    """Converts a ts object into a dataframe containing lags

    Args:
        time_series (ts or array-type): The object to transform. Can either be a ts object or an array-type containing exogenus features such as seasonality dummies
        lags (int, optional): How many lags to include. If no value is given, will create 2*seasonal period lags. Defaults to 'default'.
        remove_zeros (bool, optional): Whether to remove observations with no lag values. Defaults to True.

    Raises:
        TypeError: [description]
    """


    n = len(time_series)
    data = time_series.data
    # If lags == default I am getting two full periods
    if lags == 'default':
        try:
            lags = int(2 * get_numeric_frequency(time_series))
        except:
            raise TypeError('Lags cannot be created. Please set a manual number for the lags.')
    # Instatiating the lagged time series
    x = np.zeros((n, lags + 1))
    # Filling the lagged matrix
    for i in range((lags + 1)):
        x[i:n,i] = data[0:(n-i)]
    # Renaming: y is for the the "true" value
    cols = ["lag" + str(i) for i in range(1,(lags + 1))]
    cols = ["y"] + cols
    x = pd.DataFrame(x, columns = cols)
    #Also removing values with zeros
    if remove_zeros:
        x = x.iloc[lags:,:]
    return(x) 


def rfe_lag_selection(lagged_ts, model, features_frac):
    """Performs lag selection using the rfe method. 
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

    Args:
        lagged_ts (dataframe): The dataframe with the lag features generated through the create_lags function
        model (str): The model to perform stepwise lag selection. It is recommended to use the final model
        features_frac (float): The fraction of the total number of lags to include 

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    # Splitting into target and features

    # Getting the number of features to return
    # Only getting the integer part 
    # -1 because one is the target 
    n_features = int((lagged_ts.shape[1] - 1) * features_frac)

    cols = lagged_ts.columns.values
    target_cols = cols[cols == 'y']
    feature_cols = cols[cols != 'y']

    features = lagged_ts[feature_cols].values 
    target = lagged_ts[target_cols].values

    if model == 'lgbm':
        estimator = lgb.LGBMRegressor()
    elif model == 'xgboost':
        estimator = xgb.XGBRegressor()
    elif model == 'rf':
        estimator = RandomForestRegressor()
    #elif model =='svm': for not it is not supported
    #    estimator = SVR()
    elif model == 'lr':
        estimator = LR()
    else:
        raise TypeError('Wrong model selection. Supported models are lgbm, xgboost, rf (RandomForest), svm and lr (LinearRegression)')

    #Preparing and fitting the rfe model:
    rfe = RFE(estimator, n_features_to_select = n_features)
    fit = rfe.fit(features, target.ravel())



    # Getting the indices of the returned values
    idx = np.where(fit.support_ == True)

    # Gettign the columns and removing the first column cuz it will always be there
    cols = lagged_ts.columns.values[1:]
    new_cols = np.take(cols, idx)[0]
    # Inserting the target value y
    new_cols = np.insert(new_cols, 0, 'y')
    return new_cols



def pacf_lag_selection(lagged_ts, features_frac):
    """Performs lag selection using the pacf method as defined in #https://otexts.com/fpp3/non-seasonal-arima.html

    Args:
        lagged_ts (dataframe): The dataframe with the lag features generated through the create_lags function
        features_frac (float): The fraction of the total number of lags to include 

    Returns:
        [type]: [description]
    """


    # Getting the number of features to return
    # Only getting the integer part 
    # -1 because one is the target 
    n_features = int((lagged_ts.shape[1] - 1) * features_frac)
    
    # Taking the data and the sorted values
    data = lagged_ts['y'].values
    total_lags = lagged_ts.shape[1] - 1
    # Running the pacf test
    pacf_results = pacf(data, total_lags)
    # Removing the first because it is the original value
    pacf_results = pacf_results[1:]
    # Sorting based on the absolute value 
    pacf_results = np.argsort([abs(value) for value in pacf_results])[::-1]
    # Getting the first n results
    pacf_results = pacf_results[:n_features]    
    
    # taking the cols
    cols = lagged_ts.columns.values[1:]
    new_cols = np.take(cols, pacf_results)
    new_cols = np.insert(new_cols, 0, 'y')
    return new_cols

# The input is the lagged dataframe created using the create_lags function
# No exogenus features are added here, we just select the lags

# Model is used for the rfe method and is the model used to follow the recursive feature elimination
# Default estimator is LGBM. Also supports XGBoost, RF, linear regression, SVM

# n_features -> the fraction of features to return (Number between 0 and 1). Default is 0.6. That means from 10, returns 6
# the variable is tunable!! One can get performance improvement by lowering the number of features
def lag_selection(lagged_ts, method = 'rfe', comb = None, model = 'lgbm', features_frac = 0.6 ):
    """Performs lag selectio given a dataframe with lags as features

    Args:
        lagged_ts (dataframe): The dataframe with the lag features generated through the create_lags function
        method (str, optional): The method to perform lag selection. Supporting rfe, pacf and comb. Defaults to 'rfe'.
        comb (str, optional): The method to combine lags from both methods. Supports union and intersect. Defaults to None.
        model (str): The model to perform stepwise lag selection using the rfe method. It is recommended to use the final model
        features_frac (float): The fraction of the total number of lags to include

    Raises:
        TypeError: [description]
        TypeError: [description]

    Returns:
        Dataframe: A dataframe with the selected lags as features
    """

    if method == 'rfe':
        new_cols = rfe_lag_selection(lagged_ts, model,features_frac )
    elif method == 'pacf':
        new_cols = pacf_lag_selection(lagged_ts, features_frac)
    elif method == 'comb':
        rfe_lags = rfe_lag_selection(lagged_ts, model,features_frac )
        pacf_lags = pacf_lag_selection(lagged_ts, features_frac)
        # If we take the union:
        if comb == 'union':
            new_cols = np.union1d(rfe_lags, pacf_lags)
        elif comb == 'intersect':
            new_cols = np.intersect1d(rfe_lags, pacf_lags)
        else:
            raise TypeError('Pick another combination method. Supported methods are union and intersect ')
    else:
        raise TypeError('Pick another method. Supported methods are rfe, pacf and comb')
    
    new_lagged_ts = lagged_ts[new_cols]
    return new_lagged_ts