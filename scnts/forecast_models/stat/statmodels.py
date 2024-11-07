from pandas.core.indexes.datetimes import date_range
from scnts.forecast_models.stat.rconverters import ts_converter,initiate_r, infer_frequency
from scnts.TS.ts_class import ts
from scnts.forecast_models.stat.stat_base_model import StatisticalBaseModel, statistical_forecast

import numpy as np
import pandas as pd

import rpy2.robjects.packages as rpackages
from rpy2 import rinterface, robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri


initiate_r()
forecast_r = importr('forecast')


# The ARIMA predictor 
# Defined similarly with the ETS 
class ARIMA(StatisticalBaseModel):
    # Documentation
    # https://www.rdocumentation.org/packages/forecast/versions/8.15/topics/auto.arima
    # As with ETS at the moment we only support the automatic fitting using the AICc
    def __init__(self, *args, **kwargs) : # These are for the converter
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs


    def fit(self, ts, first = True, day = False, week = False, year = False, true_ts = 'Not'):
        # Taking the original data
        self.ts_data = ts.ts
        self.ts = ts
        self.true_ts = true_ts
        # Converting it into an r object
        self.fitted = ts_converter(self.ts, first = True, day = False, week = False, year = False)
        # Fitting the model
        self.model = forecast_r.auto_arima(self.fitted, *self.args, **self.kwargs)

    # The other functions are inheritted



class ETS(StatisticalBaseModel):
    # Accepts a ts object and builds a predictors
    #https://otexts.com/fpp2/estimation-and-model-selection.html
    
    def __init__(self, *args, **kwargs) : # These are for the converter
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def fit(self, ts, first = True, day = False, week = False, year = False,
     true_ts = 'Not'): # user can provide the test set as well.. for evaluation 
        
        # Taking the original data
        self.ts_data = ts.ts
        self.original_ts = ts
        self.true_ts = true_ts
        # Converting it into an r object
        self.fitted = ts_converter(self.original_ts, first = True, day = False, week = False, year = False)
        # Fitting the model
        self.model = forecast_r.ets(self.fitted, *self.args, **self.kwargs)



# The naive predictior
# In simple terms, the random walk with lag equal to 1
class Naive(StatisticalBaseModel):
    # Documentation:
    #https://github.com/robjhyndman/forecast/blob/038c5036441f7f96e22406203f726988fdcb35cb/R/naive.R
    def __init__(self, lag = 1, drift = False, *args, **kwargs) : # These are for the converter
        super().__init__(*args, **kwargs)
        self.lag = lag
        self.drift = drift
        self.args = args
        self.kwargs = kwargs

    def fit(self, ts, first = True, day = False, week = False, year = False,
     true_ts = 'Not'): # user can provide the test set as well.. for evaluation 
        
        # Taking the original data
        self.ts_data = ts.ts
        self.ts = ts
        self.true_ts = true_ts

        # Converting it into an r object
        self.fitted = ts_converter(self.ts, first = True, day = False, week = False, year = False)
        # Fitting the model
        self.model = forecast_r.lagwalk(self.fitted, lag = self.lag, drift = self.drift, *self.args, **self.kwargs)



# The Seasonal naive predictor
class SNaive(StatisticalBaseModel):
    # Domentantion:
    #https://github.com/robjhyndman/forecast/blob/038c5036441f7f96e22406203f726988fdcb35cb/R/naive.R
    def __init__(self, drift = False, *args, **kwargs) : # These are for the converter
        super().__init__(*args, **kwargs)
        self.drift = drift
        self.args = args
        self.kwargs = kwargs

    def fit(self, ts, first = True, day = False, week = False, year = False,
     true_ts = 'Not'):
        # Taking the original data
        self.ts_data = ts.ts
        self.ts = ts
        self.true_ts = true_ts

        self.freq = infer_frequency(ts, True, False, False, False)[0]
        self.fitted = ts_converter(self.ts, first = True, day = False, week = False, year = False)
        # Defining the model
        self.model = forecast_r.lagwalk(self.fitted, lag = self.freq, drift = self.drift, *self.args, **self.kwargs)