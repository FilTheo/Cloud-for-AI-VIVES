from modules.utils.metrics import (
    rmsse,
    bias,
    mae,
)
import numpy as np
import json
import boto3
from botocore.exceptions import ClientError

# Initial configurations
# Initial configurations
DEMO_FORECAST_CONFIGURATIONS = {"h": 7, 
                                "freq": "D", 
                                "n_jobs": 1, 
                                "holdout": False,
                                "cv": 0, 
                                "remove_weekends": False, 
                                "seasonal_length": 7,
                                "item_time_covars": ['sell_price', 'event_type_1', 'event_type_2','snap'],
                                'univariate_columns': ['unique_id', 'date', 'y'],
                                'selected_features': ['unique_id', 'date', 'y'],
                                'static_features':[]
                                }
DEMO_MODEL_CONFIGURATIONS = {
    "model": "lgbm",
    "lags": 7,
    "loss": "mse",
    #"seasonal_features":["day_of_week", 'day'],
    #"seasonal_features":["day_of_week"],
    #"lag_transforms":lag_transformation,
    #"transformations":[LocalStandardScaler()],
    "benchmark_models": ["SNaive", "SeasonalWindowAverage"],
    "fit_benchmarks": True,
    #"params": params,
        #{"n_estimators":900,
            #"learning_rate" : 0.114,
            #"max_depth": 15,
            #"num_leaves": 940,
            #"feature_fraction": 1,
            #"max_bin": 255,
    #   }

    }

DEMO_BENCHMARK_CONFIGURATIONS = {"fit_benchmarks": True, "benchmark_models": ["SNaive"]}

DEMO_EVALUATION_CONFIGURATIONS = {
    "metrics": [rmsse, mae, bias],
    "save_path": "/home/filtheo/tetra_product/static/plots",
    # "MINIMUM_TRAILING_ZEROS": 40,  # I would go for 35 but let it forecast somet
}


DEMO_AWS_LOAD_CONFIGURATIONS = {
    "bucket_name": "tetra-forecasting",
    "new_file_name": "week_1_test_beta.csv",
    "original_file_name": "week_1_train_beta.csv",
    "original_forecast_name": "week_1_forecasts_beta.csv",
    "forecast_path": "data/weekly_forecasts",
    "bucket_path": "data/weekly_updates",
    "plot_filenames": [
        "total_true_predicted_plot",
        "total_and_average_bias",
        "error_histogram",
    ],
}


FEATURE_ENGINEERING_CONFIG = {
    "n_loss": 2,
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
    "covariates": ["sell_price", "event_type_1", "event_type_2", "snap"],
    "n_seasonal": 2,
    "seasonal_features": ["day_of_week", "month", "year"],
    "n_lags": 2,
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
