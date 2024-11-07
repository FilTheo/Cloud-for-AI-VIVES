from scnts.local_preprocess.lags import create_lags
from scnts.TS.ts_class import *
from scnts.local_preprocess.preparets import prepare_ts
from scnts.forecast_models.ml.forecastingutils import *
import pandas as pd

from scnts.forecast_models.stat.statmodels import ETS

from scnts.forecast_models.ml.tuneparameters import hypertuner

from scnts.forecast_models.ml.mlmodels import LGBM
from skopt.space import Real, Categorical, Integer


from scnts.forecast_models.forecastclass.compare import compare_forecasts

df = pd.read_csv('air.csv').set_index('Month')
# true



# Keep last 12 days for testing
train = df.iloc[:-12]
test = df.iloc[-12:]
ts_df = ts(train)
test_df = ts(test)
df = ts(df)


#lagged_ts = create_lags(ts_df, lags = 'default', remove_zeros= True)

ts_ready = prepare_ts(ts_df, log_variance = False, stationarity = True, normalization = True,
 normalize_method = 'scaler', 
              seasonal_features = True, total_lags = 'default', seasonality_method = 'fourier', dummies_frequency = 'default',
            select_lags = True, lag_selection_method = 'rfe', lag_model = 'lgbm',features_frac = 0.6) #lag selection 

# Define the model
model = LGBM()

# fit the model
model.fit(ts_ready, ts_df)

#forecast
fc = model.forecast(h = 12, reverse_forecast= True, PI = True)

# Adding the real data
fc.add_real(test_df)

# Getting a summary
summary = fc.summary()

#getting the evaluation
eval = fc.evaluate()

#getting a plot
fc.plot()

# Getting a new statistical forecast to compare
ets = ETS()

# Fitting
# Here i added the real data in the fit
ets.fit(ts = ts_df, true_ts = df)

# Forecasting
ets.forecast(12)

# going for evaluation
compare = compare_forecasts([ets,fc])

# Plotting comparisons
compare.plot()

# Getting some comparisons
evals = compare.summary()
