from scnts.local_preprocess.lags import create_lags
from scnts.TS.ts_class import *
from scnts.local_preprocess.preparets import prepare_ts
from scnts.forecast_models.ml.forecastingutils import *
import pandas as pd

from scnts.forecast_models.ml.tuneparameters import hypertuner

from scnts.forecast_models.ml.mlmodels import LGBM
from skopt.space import Real, Categorical, Integer


df = pd.read_csv('air.csv').set_index('Month')
ts_df = ts(df)

#lagged_ts = create_lags(ts_df, lags = 'default', remove_zeros= True)

ts_ready = prepare_ts(ts_df, log_variance = False, stationarity = True, normalization = True,
 normalize_method = 'scaler', 
              seasonal_features = True, total_lags = 'default', seasonality_method = 'fourier', dummies_frequency = 'default',
            select_lags = True, lag_selection_method = 'rfe', lag_model = 'lgbm',features_frac = 0.6) #lag selection 

model = LGBM()
search_space = {
        #'early_stopping_rounds': [10],
        "max_depth": Integer(5, 20), # values of max_depth are integers from 6 to 20
        "num_leaves" : Integer(20, 60),
        "min_child_samples " : Integer(2, 20),
        "min_split_gain " : Real(0.001, 0.1),
        "n_estimators": Integer(100, 500),
        'feature_fraction': Real(0.1, 0.9),
        'bagging_fraction': Real(0.8, 1),
        'min_child_weight': Integer(5, 50),

    }

param_grid = {'bootstrap': [True,False],
     'max_depth': [6, 10 , 12,1  ],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [3,5, 6],
     'min_samples_split': [4,5, 6],
     'n_estimators': [100, 250, 300],
     'feature_fraction': [0.1,0.9,0.7]
    }

# converts a search space into the correct format is bayesian is picked


new_model = hypertuner(ts_ready, model, 'bayesian',  parameter_list = param_grid, cv = 2, itterations = 10, 
                       print_results = True, return_best = True )

