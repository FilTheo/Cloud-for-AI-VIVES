# The TS class.
# Every function and method starts from objects of this class

# In simple terms, a ts object is a dataframe in the right format with specific properties
# There are many ways to convert something into a ts object by passing different arguments

from scnts.TS.ts_utils import *


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math

# There are multiple options for defining an object of this class.
# It is based on what the user provides.

# Option 1: An array, the frequency
#           In this case I will add todays date as the end date and based on the lenght will add the start date
# Option 2: An array, the frequency the end date
#           In this case I will use the frequency and the length to get the start date
# Option 3: An array, the frequency the start date
#           In this case I will add the frequency and the start to  get the end date
# Option 4: An array, the frequency the end & the start
#           Using the start date and the frequency
# Option 5: An array, the start and the end
#           Here will derive the frequency from the length -> Might raise exception if not a good number is raised
# Option 6: The datetime range 
#           Here I am passing the format directly
# Option 7: A dataframe object with the index in the right format
#           Simply making the transformation

class ts(object):
    
    def __init__(self, data, date_range = [], frequency = None, start = None, end = None):
        """The main object of shats

        Args:
            data (array-type): The data to convert into time series
            date_range (list, optional): A list including the dates for the ts object. Defaults to [].
            frequency (str, optional): The frequency for the passed time series. Defaults to None.
            start (list or datetime, optional): The start date. It can be a list like [year, month, day, hour, ...]. Or a datetime object. Defaults to None.
            end (list or datetime, optional): The start date. It can be a list like [year, month, day, hour, ...]. Or a datetime object. Defaults to None.

        Raises:
            AttributeError: [description]
            TypeError: [description]
            TypeError: [description]
        """


  
        # Checking if the input is in the appropriate format and passing
        check_input(data)
        self.data = data
        
        # If it is in the right format for converting
        # if it is a dataframe with the index being either a periodindex or a datetime index
        if isinstance(data, pd.core.frame.DataFrame):
            if (isinstance(data.index, pd.core.indexes.period.PeriodIndex) or isinstance(data.index, pd.core.indexes.period.DatetimeIndex)): 
                date_range = data.index
                data = data.iloc[:,0].values
                self.data = data
                # Using the .freq method for periodindex objects
                self.frequency = get_freq(date_range)
            
            # if the index is in the right format but not as a datetime but as either as a float or a string
            # fix that -> when index is 
            elif (isinstance(data.index, pd.core.indexes.base.Index)): 
                if all((isinstance(data_i, float) or isinstance(data_i, int) or (isinstance(data_i, str))) for data_i in data.index.values):
                    # Trying to convert to datetimeindex
                    try:
                        date_range = pd.DatetimeIndex(data.index)
                    except:
                        raise AttributeError('Index could not be converted to datetime.')
                    
                    data = data.iloc[:,0].values
                    self.data = data
                    # Using the .freq method for periodindex objects

                    self.frequency = get_freq(date_range)

        # If the user has provided the date range
        elif len(date_range) != 0:
            # Checking if it is in the right format and converts to datetime
            date_range = check_date_range(date_range)
            
            # Extracting the frequency the first and the end date
            self.frequency = get_freq(date_range)
        # If the users has not provided the date range then it is infered from frequency, start and end date.
        else:
            
            # Checking if frequency is in the right format (unless no end and start date are given)
            if not isinstance(frequency, str) and not frequency == None :
                raise TypeError("Inappropriate frequency format. Use the following format: Period + Frequency. For example '5M' for 5 Months period. Allowed frequencies are: A (Annual), Q (Quarter), M (Monthly), W (Weekly), D (Daily), H (Hourly)")
            

            # Checking if start and end date are in the correct format
            if start != None and end != None:
                # Checking and converting
                start, end, only_date = check_dates(start, end)
                # If I have the frequency, the start and the end using frequency with the start.
                # Otherwise the returned date range is not correct 
                
                if frequency != None:
                    date_range = get_date_range(data, start = start, frequency = frequency)
                    self.frequency = frequency
                date_range = get_date_range(data, start = start, end = end)
                
                # If we have only date removing the hour and minutes from the date_range
                if only_date:
                    date_range = date_range.strftime('%Y-%m-%d')
                    
            # If the above condition is not satisfied  
            # In other words If one of two (or both) are None make sure that frequency is given
            elif frequency == None:
                raise TypeError('Inappropriate input format. Frequency should provided without end and start date')

            else:   # if we have frequency and either the start, the end or not any of two
                    # if both the start and the end are None then get the current date as the end date
                
                # First assigning frequency 
                self.frequency = frequency
                if start == None and end == None:
                    current_datetime = datetime.datetime.now()
                    date_range = get_date_range(data, frequency = frequency, end = current_datetime )
                    
                # if the start date exists 
                elif start != None:
                    start = convert_to_datetime(start)
                    date_range = get_date_range(data, frequency = frequency, start = start)
                
                # if the end date exists    
                elif end != None:
                    end = convert_to_datetime(end)
                    date_range = get_date_range(data, frequency = frequency, end = end)
                    

         #converting to datetime index
        if not isinstance(date_range, pd.core.indexes.period.PeriodIndex) and not isinstance(date_range, pd.core.indexes.period.DatetimeIndex):
                date_range = pd.DatetimeIndex(date_range)
         
        # Assigning frequency (if possible) if it has not already.
        if frequency == None:
            self.frequency = get_freq(date_range)
                    
        # Removing the hour/minutes from the index for better presentation
        
        # if we cannot infer the Frequency
        if self.frequency == None:
            # People might have ignored the warnings
            #warnings.warn("Frequency could not be infered.")
            print(f'Warning. Frequency could not be infered and is assigned as None.')
        else:
            #if type(self.frequency) == pd.tseries.offsets:
            #print(type(self.frequency))
            if 'H' in self.frequency:
                pass
            else:
                date_range = date_range.strftime('%Y-%m-%d')
                date_range = pd.DatetimeIndex(date_range)
        
        # Finalizing the calls
        self.date_range = date_range
        self.start = date_range[0]
        self.end = date_range[-1]
        self.length = len(data)
        
        # The final format is the dataframe
        self.ts = pd.DataFrame(data = data, index = date_range, columns = ['x'])
    
    # The call function will return the dataframe uppon calling
    def __call__(self):
        return self.ts
    # Some extra helpfull functions
    def __len__(self):
        return self.length
    
    def ts(self):
        return self().ts
    
    def head(self):
        return self.ts.head()
    
    def dates(self):
        return self.date_range
    
    def values(self):
        return self.data
    
    # Will modify this function to make it prettier
    def plot(self, grid = True):
        
        fig = plt.figure(figsize = (14,7))
        ax = fig.add_subplot(1, 1, 1)
        

        ax.plot(self.date_range, self.data,  color = 'black', linewidth = 1.5)
        
        # Editting the figure
        # The higher the number the gray_scale gets the more white the background get
        gray_scale = 0.93
        ax.set_facecolor((gray_scale, gray_scale, gray_scale))
        
        # The total number for xticks
        # dividing the length with 10 and rounding up 
        periods = int(math.ceil(len(self)/10))
        ax.xaxis.set_major_locator(plt.MaxNLocator(periods))
        
        if grid:
            ax.grid(linestyle='-', color = 'w', linewidth = 2)
        #ax.legend(loc = 2)
        
        #plt.grid(grid)
        plt.show()
        
        
    def summary(self):
        return self.ts.describe().T
    
    # Decompose!
    # Using the stable STL decomposition on statsmodels with some modifications
    # Documentation: https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html
    # for extra arguments check the documentation 
    def decompose(self, *args):
        if self.frequency:
            freq = get_numeric_frequency(self)
        else:
            print('Warning! Frequency could not be infered for seasonal plots.')
        data = self.data
        stl = STL(data, period = freq, *args)
        res = stl.fit()
        fig = plot_stl(res)
        #fig.set_size_inches(14, 8)
        plt.show()
    
    def plot_acf(self, lags, alpha = 0.05, zero = False,  **kwargs):
        data = self.data
        plot_acf(data, lags = lags, alpha = alpha, zero = False, **kwargs)
    
    def plot_pacf(self, lags, alpha = 0.05, method = 'ywm', zero = False,  **kwargs):
        data = self.data
        plot_pacf(data, lags = lags, alpha = alpha, method = method, zero = False, **kwargs)
