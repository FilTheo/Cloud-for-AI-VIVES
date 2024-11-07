### The super class for all models


# Will include the residuals, the pi and the plot

from scnts.TS.ts_class import ts
import matplotlib.pyplot as plt
import math

class base_model(object):

    # The parent class
    def __init__(
        self,
        original_ts, #the fitted time series (or the original?)
        PI = True, # weather to return the PI
        level = 95, # the level of the PI
        ):

        self.original_ts = original_ts
        self.PI = PI
        self.level = level

    # Abstract method to fit
    def fit(self):
        
        
        pass 

    # Abstract method to forecast
    def forecast(self, *args, **kwargs):

        pass

    


