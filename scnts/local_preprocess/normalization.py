# Normalizes the given time series

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scnts.TS.ts_class import ts
from numpy import log


def normalize(time_series, method, **kwargs):
    """Standarizes or normalizes the given data

    Args:
        time_series (ts): The ts object to normalize
        method (str): Normalizatio method. Supports minmax and scaler for standard scaler

    Raises:
        AttributeError: [description]

    Returns:
        [type]: [description]
    """

    # Extracting
    dates = time_series.date_range
    data = time_series.data
    # Reshaping for the normalization
    data = data.reshape(-1,1)
    # Asserting method is correct 
    if method not in ['scaler', 'minmax']:
            raise AttributeError('Normalization method not supported. Currently supporting scaler for standar scaler and minmax for MinMaxScaling')
    # Defining the scaler based on users input 
    if method == 'minmax':
        scaler = MinMaxScaler(**kwargs)
    elif method == 'scaler':
        scaler = StandardScaler(**kwargs)
    # Normalizing
    new_data = scaler.fit_transform(data)
    # Reshaping
    new_data = new_data.reshape(-1)
    # Converting to ts
    new_ts = ts(data = new_data, date_range = dates )
    return new_ts



def log_transform(original_ts):
    """Performs a log transformation to stabilize trend-depending variance 

    Args:
        original_ts (ts): The time series object to perform the transformation

    Returns:
        [type]: [description]
    """
    data = original_ts.data
    dates = original_ts.date_range
    # Transforming 
    log_data = log(data)
    new_ts = ts(data = log_data , date_range = dates)
    return new_ts






