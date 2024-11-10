from modules.utils.metrics import (
    rmsse,
    bias,
    mae,
)
import numpy as np
from mlforecast.lag_transforms import  SeasonalRollingMean, RollingMax, RollingMin, SeasonalRollingStd, SeasonalRollingMax, RollingMean
from mlforecast.target_transforms import LocalStandardScaler

# Initial configurations
DEMO_FORECAST_CONFIGURATIONS = {"h": 5, 
                                "freq": "B", 
                                "n_jobs": 1, 
                                "holdout": False,
                                "cv": 0, 
                                "remove_weekends": True, 
                                "seasonal_length": 5,
                                "item_covars" :['School', 'Size', 'Type', 'Customer'],
                                #"item_time_covars": ['Holiday','Meal', 'ProteinGroup', 'VegetablesGroup', 'StarchGroup'],
                                "item_time_covars": ['ProteinGroup', 'VegetablesGroup', 'StarchGroup'],
                                "static_features": ['School', 'Size', 'Type', 'Customer'],
                                'univariate_columns': ['unique_id', 'date', 'y'],
                                }
DEMO_FORECAST_CONFIGURATIONS['selected_features'] = DEMO_FORECAST_CONFIGURATIONS['univariate_columns'] + DEMO_FORECAST_CONFIGURATIONS['static_features'] + DEMO_FORECAST_CONFIGURATIONS['item_time_covars']


# Define configurations for lag transformations
lag_configs = [
    (RollingMean, 3, 5),
    (RollingMax, 1, 4),
    #(RollingMax, 4, 3),
    (RollingMin, 3, 4),
    (SeasonalRollingStd, 2, 5), #
    #(SeasonalRollingMean, 4, 4),
    (SeasonalRollingMax, 3, 5), #
    (SeasonalRollingMean, 3, 5), #
    #(SeasonalRollingMax, 3, 3),
    #(SeasonalRollingMin, 3, 1),
]


# Create lag_transformation dictionary
lag_transformation = {}
for i in range(1, max(config[2] for config in lag_configs) + 1):
    lag_transformation[i] = [
        (
            transform(window_size=window, season_length=5)
            if "Seasonal" in transform.__name__
            else transform(window_size=window)
        )
        for transform, window, max_key in lag_configs
        if i <= max_key
    ]

# Remove empty lists
lag_transformation = {k: v for k, v in lag_transformation.items() if v}
params = {
    "min_data_in_leaf": 45,
    "max_depth": 12,
    #"num_leaves": 50,
    # "bagging_freq": 7,
    # "bagging_fraction": 0.9,
    "learning_rate": 0.05,
    "n_estimators": 570,
}



DEMO_MODEL_CONFIGURATIONS = {
    "model": "lgbm",
    "lags": 15,
    "loss": "fair",
    #"seasonal_features":["day_of_week", 'day'],
    "seasonal_features":["day_of_week"],
    #"lag_transforms":lag_transformation,
    "transformations":[LocalStandardScaler()],
    "benchmark_models": ["SNaive", "SeasonalWindowAverage"],
    "fit_benchmarks": True,
    "params": params,
        #{"n_estimators":900,
            #"learning_rate" : 0.114,
            #"max_depth": 15,
            #"num_leaves": 940,
            #"feature_fraction": 1,
            #"max_bin": 255,
    #   }

    }
    

DEMO_BENCHMARK_CONFIGURATIONS = {"fit_benchmarks": True,
                                 #"benchmark_models": ["SNaive"],
                                 # "wednesdays_off_path" :"/home/filtheo/tetra_product/modules/utils/configurations/extra_days.csv",
                                 "wednesdays_off_path" :"modules/utils/configurations/extra_days.csv"
                                 }

DEMO_EVALUATION_CONFIGURATIONS = {
    "metrics": [rmsse, mae, bias],
    "save_path": "/home/filtheo/tetra_product/static/plots",
    # "MINIMUM_TRAILING_ZEROS": 40,  # I would go for 35 but let it forecast somet
}


FEATURE_ENGINEERING_CONFIG = {
    "n_loss": 3,
    "loss_candidates": [
        "mae",
        "mse",
        "rmse",
        "mape",
        "huber",
        "fair",
        "poisson",
        "tweedie",
    ],
    "n_covariates": 2,
    "covariates": ["Holiday", "ProteinGroup", "VegetablesGroup", "StarchGroup"],
    "n_seasonal": 2,
    "seasonal_features": ["day_of_week", "month", "year"],
    "n_lags": 3,
    "lag_canditates": np.arange(
        7, 7 * 5, 1
    ),  # Generates an array from 7 to 34 with step size of 1
    "sample_lags": 30,
}

HYPERPARAMETER_SEARCH_CONFIG = {
    "n_estimators": [
        120,
        300,
        10,
    ],  # Indicates a range from 120 to 300 with a step of 10
    "learning_rate": [
        0.005,
        0.2,
        0.005,
    ],  # Range from 0.005 to 0.2 with a step of 0.005
    "num_leaves": [20, 100, 2],  # Range from 20 to 100 with a step of 2
    "max_bin": [
        20,
        500,
        20,
    ],  # Range from 20 to 500 with a step of 20; note to scale down later
}

HYPERPARAMETER_TOTAL_SEARCH_CONFIG = {
    "n_estimators": 5,  # Specifies testing 5 configurations for the number of estimators
    "learning_rate": 5,  # Specifies testing 5 configurations for the learning rate
}
