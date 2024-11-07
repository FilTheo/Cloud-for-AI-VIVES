from pandas.tseries.frequencies import to_offset
import rpy2.robjects.packages as rpackages
from rpy2 import rinterface, robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri


def initiate_r():
    """ Initiates the r-python connection.

    Returns:
        [type]: [description]
    """

    # Initiating the python-r connection
    rpy2.robjects.numpy2ri.activate()
    utils = importr('utils')
    # Check if package is installed
    try:
        forecast_r = importr('forecast')
    except:
        # If not, install it
        print("The forecast package is not installed. Initiating installation")
        utils.install_packages('forecast', dependencies = True)
        forecast_r = importr('forecast')




initiate_r()
time_series_r_object = robjects.r('ts')





def infer_frequency(ts, first = True, day = False, month = False, week = False, year = False):
    """Returns the numeric frequency of a given time series

    Args:
        ts (ts): The time series object to infer the frequency from
        all_factors (bool, optional): For daily and hourly data. Returns all possible frequencies given the full period. Defaults to True.
        day (bool, optional): If ts is hourly, returns the frequency for daily period. Defaults to False.
        month (bool, optional): If ts is daily, returns the frequency for monthly period. Defaults to False.
        week (bool, optional): If ts is daily/hourly, returns the frequency for weekly period. Defaults to False.
        year (bool, optional): If ts is daily, returns the frequency for an annual period. Defaults to False.

    Returns:
        int: The numeric frequency of the given object
    """
    # Taking the frequency
    freq = ts.frequency
    # Converting to offset
    offset_freq = to_offset(freq)
    # Taking the number of periods
    period = offset_freq.n
    # Taking just the frequency 
    freq = offset_freq.name[0]
    
    #Here make the translation to numeric frequency 
    
    main_keys = ['Q', 'M', 'W','Y']
    main_vals = [4, 12, 52, 1]
    main_freq_dict = dict(zip(main_keys, main_vals))
    
    if freq in main_keys:
        # returning it in an array format to match with the equivelant if I have weekly or hourly data
        freq_numeric = [main_freq_dict[freq]]
        
    # If we have daily or hourly data I have multiple frequencies
    # For example: Days in a week, days in a month, days in a year
    # Returning them all 
    
    elif freq == 'D':
        # If the option to construct the full hierarchy is true 
        if first == True:
            freq_numeric = [7, 30, 364]
        else:
        # If not then manualy appending should have defined everything
            freq_numeric = []
            if week: freq_numeric.append(7)
            if month: freq_numeric.append(30)
            if year: freq_numeric.append(364)
    #Similarly 
    elif freq == 'H':
        if first == True:
            freq_numeric = [24, 168, 720, 8736]
        else:
            freq_numeric = []
            if day: freq_numeric.append(24)
            if week: freq_numeric.append(168)
            if month: freq_numeric.append(720)
            if year: freq_numeric.append(8736)
                
    # Taking the integer part for each period.
    # For example if I have frequency of 2M (2 months) then I have a frequency of 6 (6 2-month periods in a year)
    # If I have 5M then I have a frequency of 2 (two 5-months periods inside a year)
    freq_numeric = [int(i/period) for i in freq_numeric]
    return freq_numeric


# Convert from python ts object to r ts object

# If we have daily or hourly frequency useer has to provide which period he wants
# Otherwise the first one will be picked (weekly for day and daily for hour)
def ts_converter(ts, first = True, day = False, week = False, year = False):



    data = ts.data
    start = ts.start
    end = ts.end
    # If frequency is provided
    # If not will infer it automatically
    try:
        freq = infer_frequency(ts, True, False, False, False)[0]
    except:
        # If frequency is not provided adding it as 1
        freq = 1
        
        
   # Testing if we have hourly data
    try:
        if 'H' in ts.frequency:
            hour = True
        else:
            hour = False
    except:
        hour = False
    
    if hour == False:    
        try:
            # Extracting the time-steps from the start and end date 
            if type(start) == str:
                str_start = start
                str_end = end
            else:
                str_start = start.strftime("%Y/%m/%d")
                str_end = end.strftime("%Y/%m/%d")

            if '/' in str_start:
                start_year, start_month, start_day = str_start.split('/')
                end_year, end_month, end_day = str_end.split('/')
            else:
                start_year, start_month, start_day = str_start.split('-')
                end_year, end_month, end_day = str_end.split('/')


            # Converting the values to a ts object in R
            ts_r = time_series_r_object(data, frequency = freq, start = robjects.IntVector((start_year, start_month, start_day)),
                                           end = robjects.IntVector((end_year, end_month,end_day)))
        except:
            ts_r = time_series_r_object(data, frequency = freq)
    else:
        # Converting the values to a ts object in R
        ts_r = time_series_r_object(data, frequency = freq)
    return ts_r