# Function for normalization and a function including the complete pipeline

# Standarizes or normalizes data
# Supported methods are scaler and minmax
# Returns the normalized ts 



from scnts.TS.ts_class import ts
from scnts.local_preprocess.seasonal_features import *
from scnts.local_preprocess.stationarity import *
from scnts.local_preprocess.lags import *
from scnts.local_preprocess.normalization import *


def prepare_ts(time_series, log_variance = False, stationarity = True,
               normalization = True, normalize_method = 'scaler', # for normalization
              seasonal_features = True, seasonality_method = 'dummies', fourier_order = 1, dummies_frequency = 'default',
              total_lags = 'default' ,
            select_lags = True, lag_selection_method = 'rfe', lag_model = 'lgbm', comb_method = None, features_frac = 0.6,
            **kwargs
              ):
    """ The complete pipeline for transforming a time series into an object for machine learning training

    Args:
        time_series (ts): The ts object to prepare
        log_variance (bool, optional): Whether to stabilize the variance through a log transformation. Defaults to False.
        stationarity (bool, optional): Whether to transform the ts into stationary. Defaults to True.
        normalization (bool, optional): Whether to normalize the input data Defaults to True.
        normalize_method (str, optional): How to normalize the input data. Currently supporting minmax and scaler for standard scaling. Defaults to 'scaler'.
        seasonality_method (str, optional): Whether to include seasonality exogenus features. Supports dummies and fourier.  Defaults to 'dummies'.
        fourier_order (int, optional): If fourier is given, the order of the fourier transformation. Defaults to 1.
        dummies_frequency (str, optional): If dummies is given, the number of lags to include. Defaults to 'default'.
        total_lags (int, optional): How many lags to include on lag selection. If default is given, returns 2*seasonal period lags. Defaults to 'default'.
        select_lags (bool, optional): Whether to perform feature selection. Defaults to True.
        lag_selection_method (str, optional): If select_lags is True, choose how to perform lag selection. Supports pacf, rfe and comb. Defaults to 'rfe'.
        lag_model (str, optional): If lag_selection_method is rfe, choose the model to perform the lag selection. It is recommended to pass the final forecasting model. Defaults to 'lgbm'.
        comb_method (str, optional): If lag_selection_method is comb, choose how to combine features. Supports union and intersect . Defaults to None.
        features_frac (float, optional): The fraction of the total features to include (eg for 10 features if frac = 0.6 returns 6 lags). Defaults to 0.6.

    Raises:
        NameError: [description]

    Returns:
        [type]: A prepared dataframe object for machine learning training
    """
    
    # Step 1: Log transform
    if log_variance:
        time_series = log_transform(time_series)
    
    # Step 2: Convert to stationary
    if stationarity:
        time_series = seasonal_stationary(time_series, print_results = False)
        time_series = stationary(time_series, print_results = False)
    
    # Step 3: Normalize
    if normalization:
        time_series = normalize(time_series, normalize_method, **kwargs)
        
    # Step 4: Seasonal dummies
    if seasonal_features:
        if seasonality_method not in ['dummies', 'fourier']:
            raise NameError('Method for seasonal features not supported. Currently supporting dummies and fourier.') 
        if seasonality_method == 'fourier':
            extra_features = fourier_features(time_series, order = fourier_order)
        elif seasonality_method == 'dummies':
            extra_features = seasonality_dummies(time_series, frequency = 'default')
    
    # Step 5: Creating the lags
    # If we have the default parameter, it returns a full frequency circle
    if total_lags == 'default':
        total_lags = int(get_numeric_frequency(time_series))
        
    lagged_ts = create_lags(time_series, total_lags)
    # Also taking the first value to correctly append seasonal features if given
    first_idx = lagged_ts.index[0]
    
    # Step 6: Lag selection
    if select_lags:
        lagged_ts = lag_selection(lagged_ts, lag_selection_method, comb =  comb_method, model = lag_model, features_frac = features_frac)
    
    # If we have seasonal features appending them here
    if seasonal_features:
        # First cutting the observations removed due to creating the lags (the zeroed values)
        extra_features = extra_features.iloc[first_idx:]
        # Then appending 
        lagged_ts = pd.concat([lagged_ts, extra_features], axis = 1)
    return lagged_ts