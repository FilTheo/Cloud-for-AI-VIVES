
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
import numpy as np
from pmdarima.utils import diff_inv

from scnts.local_preprocess.lags import *
from scnts.local_preprocess.normalization import *
from scnts.local_preprocess.stationarity import *
from scnts.local_preprocess.seasonal_features import *
from scnts.local_preprocess.preparets import *



# checks what kind of transformations the user provided 
def check_transformations(original, lagged_ts):
    # Taking all posible transformations
    
    final_values = lagged_ts['y'].values
    # Taking the log transformations
    original_log = log_transform(original)
    # Taking two stationarities: One for the log transformed and one for the non-log transformed

    # For the log
    log_seas_stat, log_seasonal_tf = seasonal_stationary(original_log, return_total_differences = True, print_results = False)
    log_stat, log_stat_tf = stationary(log_seas_stat, return_total_differences = True, print_results = False)

    # For the non log
    seas_stat, seasonal_tf = seasonal_stationary(original, return_total_differences = True, print_results = False)
    stat, stat_tf = stationary(seas_stat, return_total_differences = True, print_results = False)

    # Going for normalization
    # Two types of normalization for either loged or nonlogged


    # For loged data 
    # for stationary log_data
    log_minmaxer = MinMaxScaler()
    log_standarder = StandardScaler()
    
    log_minmaxed = log_minmaxer.fit_transform(log_stat.data.reshape(-1,1)).reshape(-1)
    log_standarded = log_standarder.fit_transform(log_stat.data.reshape(-1,1)).reshape(-1)
    
    # for non stationary log data
    log_minmaxer_nonstat = MinMaxScaler()
    log_standarder_nonstat = StandardScaler()
    
    log_standarded_nonstat = log_standarder_nonstat.fit_transform(original_log.data.reshape(-1,1)).reshape(-1)
    log_minmax_nonstat = log_minmaxer_nonstat.fit_transform(original_log.data.reshape(-1,1)).reshape(-1)

    # For non logged data
    minmax = MinMaxScaler()
    standard = StandardScaler()
    
    minmaxed = minmax.fit_transform(stat.data.reshape(-1,1)).reshape(-1)
    standarded = standard.fit_transform(stat.data.reshape(-1,1)).reshape(-1)
    # For non stat 
    
    minmax_nonstat = MinMaxScaler()
    standard_nonstat = StandardScaler()
    
    minmaxed_nonstat = minmax_nonstat.fit_transform(original.data.reshape(-1,1)).reshape(-1)
    standarded_nonstat = standard_nonstat.fit_transform(original.data.reshape(-1,1)).reshape(-1)

    # Checking if the final values are a subset of any of the transformed sets 

    # for non_logged data
    # Non log and stationary
    if set(final_values) <= set(standarded):
        normalizer, stationarity, log = standard, True, False
    elif set(final_values) <= set(minmaxed):
            normalizer, stationarity, log = minmax, True, False
    # Non log and non stationary 
    elif set(final_values) <= set(minmaxed_nonstat):
            normalizer, stationarity, log = minmax_nonstat, False, False
    elif set(final_values) <= set(standarded_nonstat):
            normalizer, stationarity, log = standard_nonstat, False, False

    # For non normalized data
    # Non normalized stationary 
    elif set(final_values) <= set(stat.data):
            normalizer, stationarity, log = None, True, False
    # non normalized non stationary (aka original)
    elif set(final_values) <= set(original.data):
            normalizer, stationarity, log = None, False, False        

    # For log data
    # for stationary log data
    elif set(final_values) <= set(log_standarded):
        normalizer, stationarity, log = log_standarder, True, True
    elif set(final_values) <= set(log_minmaxed):
        normalizer, stationarity, log = log_minmaxer, True, True
    # for non stationary log data 
    elif set(final_values) <= set(log_standarded_nonstat):
        normalizer, stationarity, log = log_standarder_nonstat, False, True
    elif set(final_values) <= set(log_minmax_nonstat):
        normalizer, stationarity, log = log_minmaxer_nonstat, False, True

    # For non normalized log data
    # Non normalized log stationary
    elif set(final_values) <= set(log_stat.data):
            normalizer, stationarity, log = None, True, True
    # non normalized non stationary (aka original)
    elif set(final_values) <= set(original_log.data):
            normalizer, stationarity, log = None, False, True        
    
    else:
        normalizer, stationarity, log = None, None, None
    
    # Also returns the number of seasonal differences
    if stationarity:
        # if we have log data 
        if log:
            total_tfs = (log_stat_tf, log_seasonal_tf)
        else:
            total_tfs = (stat_tf, seasonal_tf)
    else:
        total_tfs = (0,0)
    
    
    return normalizer, stationarity, log, total_tfs


def dif_reverse(to_reverse, start, steps):
    if type(to_reverse) == ts:
        x = to_reverse.data
    else:
        x = to_reverse
    inv = np.r_[start,x]
    inv = diff_inv(inv, lag = steps)[steps:]
    return inv

def stationary_reverse(original, to_reverse, total_tfs, log):
    stat_tfs, seasonal_tfs = total_tfs
    freq = int(get_numeric_frequency(original))
    
    if log:
        original = log_transform(original)
    # Getting the seasonaly stationary 
    seas_stat = seasonal_stationary(original, print_results = False)
    
    # Reversing the trend stationarity
    if stat_tfs >= 1:
        trend_stat_reversed = reverse_single_stationarity_step(to_reverse, seas_stat, stat_tfs, 1)
    else:
        trend_stat_reversed = seas_stat
    # Reversing the seasonal stationarity
    if seasonal_tfs >= 1 :
        stat_reversed = reverse_single_stationarity_step(trend_stat_reversed, original, seasonal_tfs, freq)
    else:
        stat_reversed = trend_stat_reversed
    return stat_reversed

def reverse_single_stationarity_step(to_reverse, original, n_diff, step):
    start_point = original
    starting_dates = original.date_range
    if type(starting_dates) != pd.core.indexes.period.PeriodIndex and type(starting_dates) != pd.core.indexes.period.DatetimeIndex:
        starting_dates = pd.DatetimeIndex(starting_dates)
    freq = original.frequency

    # Getting the starting values
    starting_vals = []
    for i in range(n_diff):
        if step == 1:
            starting_vals.append(start_point.data[0])
        else:
            starting_vals.append(start_point.data[:step])
        start_point = differences(start_point, step)
    # Reversing
    for i in reversed(range(n_diff)):
        start = starting_vals[i]
        to_reverse = dif_reverse(to_reverse, start, step)
    past_dates = len(to_reverse) - len(starting_dates)
    # adding the past values
    extra_dates = starting_dates.shift(past_dates, freq)
    new_date_range = starting_dates.union(extra_dates)
    to_return = ts(to_reverse, date_range = new_date_range)
    return to_return

