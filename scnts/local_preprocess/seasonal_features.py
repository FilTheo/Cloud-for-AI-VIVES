# adds seasonal features


from scnts.TS.ts_utils import get_numeric_frequency
from pmdarima.preprocessing import FourierFeaturizer
import pandas as pd
import numpy as np

# Hyndmans reference For fourier series
# https://otexts.com/fpp2/useful-predictors.html#useful-predictors
# Creates pairs of cosines and sines

# The order is the number of terms for cosine and sine.
# Default is 1 which return 1 term for each 
# It can exceed m/2 where m is the frequency

def fourier_features(time_series, order = 1):
    # extracting data
    data = time_series.data
    dates = time_series.date_range
    # Getting the numeric frequency
    numeric_freq = int(get_numeric_frequency(time_series))
    
    # Instatiating 
    fourier_series = FourierFeaturizer(numeric_freq, order)
    dummy, exog = fourier_series.fit_transform(data)
    # Returns a matrix with the furier features which can be appended (vstack) in a matrix ready for training
    fourier_terms = exog.values
    
    # Renaming columns
    cols = [col.replace('FOURIER_S' + str(numeric_freq), 'SIN') for col in exog.columns.values]
    cols = [col.replace('FOURIER_C' + str(numeric_freq), 'COS') for col in cols]

    fourier_terms = pd.DataFrame(data = fourier_terms, columns = cols)
    return fourier_terms


# Frequency can either be default or be given by the user in the form of a list
# for example ('M','Y') for months and years

def seasonality_dummies(time_series, frequency = 'default'):
    # Get the frequency, the dates
    freq = time_series.frequency[0]
    # To deal with frequencies in the format of 'MS', 'ME' or '2M' etc
    try:
        freq = int(freq)
        freq = time_series.frequency[1]
    except:
        pass
    
    dates = pd.DatetimeIndex(time_series.date_range)
    
    # If I need to extract seasonality on my own
    if frequency == 'default':
        # Depending on the seasonality gets the right dummies
        if freq == 'M':
            dummies = pd.get_dummies(dates.month)
        elif freq == 'A' or freq == 'Y':
            dummies = pd.get_dummies(dates.year)
        elif freq == 'W':
            dummies = pd.get_dummies(dates.week)
        elif freq == 'D':
            dummies = pd.get_dummies(dates.day)
        elif freq == 'Q':
            dummies = pd.get_dummies(dates.quarter)
        elif freq == 'H':
            dummies = pd.get_dummies(dates.hour)
        else:
            raise TypeError('Seasonality not supported. Supported frequencies are Hourly, Daily, Weekly, Monthly, Quarterly and Annualy')
        # Returns a dataframe which can be appended to the dataframe object for training
    
    #Elif seasonality is given in the form of a list
    else:
        # Instatiating 
        dummies = np.zeros([len(time_series),1])
        cols = []
        for freq in frequency:
            if freq == 'M':
                dummies_i = pd.get_dummies(dates.month)
            elif freq == 'A' or freq == 'Y':
                dummies_i = pd.get_dummies(dates.year)
            elif freq == 'W':
                dummies_i = pd.get_dummies(dates.week)
            elif freq == 'D':
                dummies_i = pd.get_dummies(dates.day)
            elif freq == 'Q':
                dummies_i = pd.get_dummies(dates.quarter)
            elif freq == 'H':
                dummies_i = pd.get_dummies(dates.hour)
            else:
                raise TypeError('Seasonality not supported. Supported frequencies are Hourly, Daily, Weekly, Monthly, Quarterly and Annualy')
            # Stacking 
            dummies = np.hstack([dummies,dummies_i.values])
            cols = np.concatenate([cols,dummies_i.columns.values])
        # Removing the first column
        dummies = dummies[:,1:]
        dummies = pd.DataFrame(dummies,columns = cols)
        
    return dummies