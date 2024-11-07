# Functions for converting to stationary 

# Will document it later

from sats.TS.ts_class import ts
from sats.TS.ts_utils import get_numeric_frequency

import statsmodels.tsa.stattools as sm
import pmdarima

# Helpfull function for the stationarity 
# Performs the differences and drops nans
def differences(data, order):
    data = data()
    return ts(data.diff(order).dropna())

# The Augmented Dickey-Fuller (ADF) test used for stationarity 
# Returns 1 if data is stationary
# 0 otherwise
def adf_test(data):
    test_values = sm.adfuller(data)
    #This means transformations
    if test_values[0] > test_values[4]['1%']:
        result = 1
    else:
        result = 0
    return result

# Checks for seasonal stationarity
# If return_total_differences = True, returns the number of differences in addition to stationary series
def seasonal_stationary(time_series, return_total_differences = False, print_results = True):
    # Initializing, gettign the numeric frequency and the vector of values
    freq = int(get_numeric_frequency(time_series))
    total_seas_diff = 0
    data = time_series.data
    # Using the Osborn, Chui, Smith, and Birchenhall (OCSB) test as defined by hyndman on nsdiff documentation
    # https://github.com/robjhyndman/forecast/blob/master/R/unitRoot.R
    # The test returns 1 if it needs a single seasonal difference and 0 otherwise
    test_result = pmdarima.arima.OCSBTest(m = freq).estimate_seasonal_differencing_term(data)
    
    # While we are getting positive results 
    while (test_result != 0 ):
        time_series = differences(time_series, freq)
        data = time_series.data
        # Increasing the number of differences
        total_seas_diff += 1
        # Testing again
        test_result = pmdarima.arima.OCSBTest(m = freq).estimate_seasonal_differencing_term(data)    
    
    if print_results:
        print(f'Total number of seasonal differences applied for seasonal stationarity: {total_seas_diff}')
    if not return_total_differences:
        return time_series
    else:
        return time_series, total_seas_diff

    
# Checks and converts to seasonal stationarity 
def stationary(time_series, return_total_differences = False, print_results = True):
    total_diff = 0
    data = time_series.data
    #Returns 1 if it is not stationary
    results = adf_test(data)
    # Similarly to before 
    while (results != 0 ):
        # Performs the differences
        time_series = differences(time_series, 1)
        data = time_series.data
        total_diff += 1 
        # Testing again
        results = adf_test(data)

    if print_results:
        print(f'Total number of differences applied for stationarity: {total_diff}')
    if not return_total_differences:
        return time_series
    else:
        return time_series, total_diff

