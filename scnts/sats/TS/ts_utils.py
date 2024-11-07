import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import pacf, acf



# Asserting that the input is in the right format
# array-type or dataframe
def check_input(x):
    if not isinstance(x, list) and not isinstance(x, np.ndarray) and not isinstance(x, pd.core.frame.DataFrame):
        # Might change the TypeError
        raise TypeError('Inappropriate input format: {} for values whereas a list or a dataframe is expected'.format(type(x)))
        
    # if it is an array (or list) all elements should be numbers
    if isinstance(x, list) or isinstance(x, np.ndarray):
        if not all(isinstance(value, int) or isinstance(value, float)  for value in x):
            raise TypeError('Inappropriate value format for elements in data. Expected numeric values')


# Converts an array of dates into a datetime object
def convert_to_datetime(x):
    
    #If the end date is in the right format return it right away
    if isinstance(x, datetime.datetime):
        return x
    # If it is str and needs converting 
    elif isinstance(x, str):
        # If it is in the correct format
        try:
            date = datetime.datetime.fromisoformat(x)
            return date
        except:
            raise TypeError('Datetime not in the right format. Convert to YYYY-MM-DD')
    
    # If it is in the format of list of structure [year, month, day, hour, minute, seconds]
    # If the array has not the full length then padding with zeros
    elif isinstance(x, list):
        # If it is over the threshold of 6
        if len(x) > 6:
            raise TypeError('Incorrect date format. Stick to the [year, month, day, hour, minutes, seconds] structure for lists')
        # If it is in the format [year, month] or [year]
        elif len(x) < 3: 
            # Padding with 1 -> for days and/or months
            # No reason to pad for hours, minutes, seconds
            x = np.pad(x, (0, 3 - len(x)), 'constant', constant_values = 1)
            year, month, day = x
            date = datetime.datetime(year, month, day)
        # no padding here
        elif len(x) == 3:
            year, month, day = x
            date = datetime.datetime(year, month, day)
        # If the user has provided the full details
        elif len(x) > 3 and len(x) <= 6:
            #Padding with zeros at the end 
            x = np.pad(x, (0, 6 - len(x)), 'constant', constant_values = 1)
            # Unzip 
            year, month, day, hour, minutes, seconds = x
            date = datetime.datetime(year, month, day, hour, minutes, seconds)
        
        return date
    # If it is not in any of these formats:
    else:
        raise TypeError('Incorrect date format.')


# A function to check the start and end date
# They should be in the right format and never start > end 
def check_dates(start, end):
    # checking if we will remove the hour, minutes from the index
    if len(start) < 4 and len(end) < 4:
        only_date = True
    else:
        only_date = False
    #Converting both to datetime
    start_date = convert_to_datetime(start)
    end_date = convert_to_datetime(end)
    
    # Check if the end date is before the start date and raise exception
    if end_date < start_date:
        raise TypeError('Incorrect date format. End date cannot be before the start date')
    return start_date, end_date, only_date

# Extract the frequency of a datetimeindex or periodindex
def get_freq(idx):
    
    if idx.freq is None:
        freq = pd.infer_freq(idx)
    elif idx.freq is not None:
        
        freq = idx.freq.name
    else:
        raise AttributeError('no discernible frequency found. Specify a frequency string with when instatiating.')
    return freq

# Checking if the date range is in the right format 
def check_date_range(date_range):
    # If it is already in the correct format
    if isinstance(date_range, pd.core.indexes.datetimes.DatetimeIndex):
        return pd.DatetimeIndex(date_range)
    
    elif isinstance(date_range, pd.core.indexes.base.Index):
        try: 
            return pd.DatetimeIndex(date_range)
        except:
            pass
    # If it is in string format
    elif all(isinstance(value, str) for value in date_range):
        # Converting to datetime range
        date_range = pd.to_datetime(date_range, infer_datetime_format = True) 
        return date_range
    # If it is neither:
    else:
        raise TypeError("Inappropriate date range format. Date range should be a DateTimeIndex object or an array of string dates")

# Function to get the date range given two out of the start, end or freq
def get_date_range(x, start = None, end = None, frequency = None):
    # Getting the total values
    total_periods = len(x)

    date_range = pd.date_range(start = start, end = end ,periods = total_periods, freq = frequency, normalize = True)
    return date_range


def get_numeric_frequency(ts):
    """Returns the frequency of a ts object in numeric format. Refer to https://otexts.com/fpp3/tsibbles.html for details

    Args:
        ts (ts): The time series in the ts format.

    Returns:
        [int]: The frequency of the time series converted from string to number.
    """
    
    keys = ['Y', 'A', 'Q', 'M', 'W', 'D', 'H']
    vals = [1, 1, 4,  12, 52, 7, 24 ]
    freq_dictionary = dict(zip(keys,vals))
    
    # Getting the period and the frequency
    period = to_offset(ts.frequency).n
    # Taking the first letter of the frequency in case we have MS for month start etc
    freq = to_offset(ts.frequency).name[0]
    # Initializing the dictionary
    numeric_freq = freq_dictionary[freq]
    # Dividing with the period: For example if I have a 2M frequency then instead of 12 months we have 6 examina
    numeric_freq = int(freq_dictionary[freq]/period)
    return numeric_freq


def plot_stl( stl, observed = True, seasonal = True, trend = True, resid = True):
    """ Returns the stl plot. 
    Reference: https://github.com/statsmodels/statsmodels/blob/7f781109b4c1270d5182f070318da5d7657dcc5e/statsmodels/tsa/seasonal.py

    Args:
        stl (STL): A statsmodels STL object
        observed (bool, optional): If to plot the observed time series. Defaults to True.
        seasonal (bool, optional): If to plot the seasonality of the observed time series. Defaults to True.
        trend (bool, optional): If to plot the trend of the observed time series. Defaults to True.
        resid (bool, optional): If to plot the residuals of the observed time series.. Defaults to True.

    Returns:
        [figure]: Plots a pyplot figure.
    """
    from pandas.plotting import register_matplotlib_converters

    from statsmodels.graphics.utils import _import_mpl

    register_matplotlib_converters()
    
    series = [(stl._observed, "Observed")] if observed else []
    series += [(stl.trend, "trend")] if trend else []
    series += [(stl.seasonal, "seasonal")] if seasonal else []
    series += [(stl.resid, "residual")] if resid else []

    if isinstance(stl._observed, (pd.DataFrame, pd.Series)):
        nobs = stl._observed.shape[0]
        xlim = stl._observed.index[0], stl._observed.index[nobs - 1]
    else:
        xlim = (0, stl._observed.shape[0] - 1)
    
    gray_scale = 0.93
    
    fig, axs = plt.subplots(len(series), 1, figsize = (14,8))
    
    for i, (ax, (series, def_name)) in enumerate(zip(axs, series)):
        if def_name != "residual":
            ax.plot(series, color = 'black')
            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
            ax.grid(linestyle='-', color = 'w', linewidth = 2)
        else:
            ax.plot(series, marker = "o", linestyle="none", color = 'black')
            ax.plot(xlim, (0, 0), color = "#000000", zorder=-3)
            ax.set_facecolor((gray_scale, gray_scale, gray_scale))
            ax.grid(linestyle='-', color = 'w', linewidth = 2)
        name = getattr(series, "name", def_name)
        if def_name != "Observed":
            name = name.capitalize()
        title = ax.set_ylabel
        title(name)
        ax.set_xlim(xlim)
    fig.subplots_adjust(hspace = .0001)
    fig.tight_layout()
    return fig

def plot_acf(x, lags, alpha = 0.05, zero = False, **kwargs):
    """Returns the acf plot of the given time series.
    References: 
    https://www.statsmodels.org/dev/_modules/statsmodels/graphics/tsaplots.html#plot_pacf
    https://github.com/statsmodels/statsmodels/blob/fcc8b500fd893923311bd3a6cffe1bc6bff22a1f/statsmodels/graphics/tsaplots.py#L32
    
    Args:
        x (array-type): The time series data 
        lags (int): The total number of lags to include
        alpha (float, optional): The covarage probability. Defaults to 0.05.
        zero (bool, optional): Whether to include the original observation for lag = 0. Defaults to False.
    """
    acf_ = acf(x, nlags = lags, alpha = alpha, **kwargs)
    
    # splitting acf and the intervals
    if alpha is not None:
        acf_x, confint = acf_[:2]
    
    if not zero:
        acf_x = acf_x[1:]
        confint = confint[1:]
    else:
        lags = lags + 1
    lags_x = np.arange(0, lags)
    fig = plt.figure(figsize = (10,5))
    ax = plt.subplot(1,1,1)
    
    ax.vlines(lags_x, [0], acf_x)
    ax.axhline()
    ax.margins(0.05)
    ax.set_title('Autocorrelation')
    ax.plot(lags_x, acf_x, marker = 'o', markersize = 5, markerfacecolor = 'red', markeredgecolor = 'red')

    #ax.set_ylim(-1, 1)
    # Setting the limits
    ax.set_ylim(
        1.25 * np.minimum(min(acf_x), min(confint[:, 0] - acf_x)),
        1.25 * np.maximum(max(acf_x), max(confint[:, 1] - acf_x)))

    lags_x[0] -= 0.5
    lags_x[-1] += 0.5
    ax.fill_between(lags_x, confint[:, 0] - acf_x, confint[:, 1] - acf_x, alpha=0.25)

    #gray_scale = 0.93
    #ax.set_facecolor((gray_scale, gray_scale, gray_scale))
    ax.grid()
    plt.show()
    
    
def plot_pacf(x, lags, alpha = 0.05, method = 'ywm', zero = False, **kwargs):
    """Returns the pacf plot of the given time series.
    References: 
    https://www.statsmodels.org/dev/_modules/statsmodels/graphics/tsaplots.html#plot_pacf
    https://github.com/statsmodels/statsmodels/blob/fcc8b500fd893923311bd3a6cffe1bc6bff22a1f/statsmodels/graphics/tsaplots.py#L32
    
    Args:
        x (array-type): The time series data 
        lags (int): The total number of lags to include
        alpha (float, optional): The covarage probability. Defaults to 0.05.
        method (str, optional): The method for estiamating the partial autocorrelation coefficients. Defaults to 'ywm'.
        zero (bool, optional): Whether to include the original observation for lag = 0. Defaults to False.
    """
    
    pacf_x = pacf(x, nlags = lags, alpha = alpha, method = method)
    # splitting acf and the intervals
    if alpha is not None:
        pacf_x, confint = pacf_x[:2]

    if not zero:
        pacf_x = pacf_x[1:]
        confint = confint[1:]
    else:
        lags = lags + 1

    lags_x = np.arange(0, lags)
    fig = plt.figure(figsize = (10,5))
    ax = plt.subplot(1,1,1)

    ax.vlines(lags_x, [0], pacf_x)
    ax.axhline()
    ax.margins(0.05)
    ax.set_title('Partial Autocorrelation')
    ax.plot(lags_x, pacf_x, marker = 'o', markersize = 5, markerfacecolor = 'red', markeredgecolor = 'red')



    # Setting the limits
    ax.set_ylim(
        1.25 * np.minimum(min(pacf_x), min(confint[:, 0] - pacf_x)),
        1.25 * np.maximum(max(pacf_x), max(confint[:, 1] - pacf_x)))

    lags_x[0] -= 0.5
    lags_x[-1] += 0.5
    ax.fill_between(lags_x, confint[:, 0] - pacf_x, confint[:, 1] - pacf_x, alpha=0.25)

    #gray_scale = 0.93
    #ax.set_facecolor((gray_scale, gray_scale, gray_scale))
    ax.grid()
    plt.show()


