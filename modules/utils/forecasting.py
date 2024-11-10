import pandas as pd
#import cloudpickle
#import fsspec
from modules.utils.utils import statsforecast_forecast_format, get_numeric_frequency, transaction_df

from mlforecast import MLForecast
from mlforecast.core import TimeSeries
from mlforecast.lag_transforms import BaseLagTransform
import utilsforecast.processing as ufp

import numpy as np

from statsforecast import StatsForecast

import warnings
#from statsmodels.tsa.stattools import acf
#from statsmodels.tsa.exponential_smoothing.ets import ETSModel

#import dask.dataframe as dd
#from dask.distributed import Client
#from fugue_dask import DaskExecutionEngine

from statsforecast.models import (
    AutoETS,
    AutoARIMA,
    Naive,
    SeasonalNaive,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    WindowAverage,
    SeasonalWindowAverage,
    AutoCES,
    AutoTheta,
)
from modules.utils.utils import check_covars
from logs.logging_config import setup_logger
import psutil

logger = setup_logger(__name__, 'forecasting.log')

# Deal with the stupid slice warning.
pd.options.mode.chained_assignment = None


class GlobalForecaster(object):
    def __init__(
        self,
        models,
        lags,
        freq,
        seasonal_length=None,
        lag_transforms=None,
        transformations=None,
        date_features=None,
        n_jobs=1,
        hyperparams_cv = False
    ):
        self.freq = freq
        self.seasonal_length = seasonal_length or get_numeric_frequency(freq)
        if isinstance(self.seasonal_length, list):
            self.seasonal_length = self.seasonal_length[0]
        self.n_jobs = n_jobs
        self.models = models
        self.lag_transforms = expand_lag_transforms(lag_transforms) if lag_transforms else None
        self.transformations = transformations
        self.date_features = date_features
        self.lags = lags
        self.model_names = models.keys()
        self.target_col = "y"
        self.id_col = "unique_id"
        self.time_col = "ds"

        self.forecaster = MLForecast(
            models=self.models,
            freq=self.freq,
            lags=self.lags,
            lag_transforms=lag_transforms,
            target_transforms=self.transformations,
            date_features=self.date_features,
            num_threads=self.n_jobs,
        )
        self.hyperparams_cv = hyperparams_cv

    def _fit(
        self,
        df,
        filter_zero_lags=True,
        future_covars=None,
    ):

        # Move these to normal fit
        # pass new_df the temp_df

        # Prepare the preprocessed df
        df = self.forecaster.preprocess(df)
        self.for_insample = df

        if filter_zero_lags:
            df = filter_identical_lag_rows(df)

        # Extra preprocessing on the feature df can be added here
        if future_covars is not None:
            df = pd.merge(
                future_covars, df, how="right", on=["unique_id", "ds"]
            )

            # self.feature_df = merge_future_covars(self.feature_df, future_covars, merge_on)

        # Split the data
        # self.x, self.y = self.forecaster._extract_X_y(self.feature_df, self.target_col)

        # Using a manual split to keep the covariates
        # self.x = self.feature_df.drop(columns = [self.target_col, self.id_col, self.time_col])
        # self.y = self.feature_df[self.target_col].values
        self.x, self.y = split_X_y(
            df, self.target_col, self.id_col, self.time_col
        )

        # Fit the models
        self.forecaster.fit_models(self.x, self.y)

        return self  # just to print here

    def fit(
        self,
        df,  # df in transactional format
        # prediction_intervals=...,  # Will
        static_features=None,  # static covariates
        merge_on=None,  # The column to merge the static covariates on
        fill_covars=None,  # Fill the missing values in the static covariates
        dropna=True,  # Drop missing rows
        filter_missing=None,  # Filter out items with less than a number of non-zero observations
        filter_zero_lags=None,  # Filter out windows with consecutive zeros
        # PIs should be called on the fit because we need to get a validation set before fitting!
        calculate_intervals=False,
        max_horizon=None,
        pi_type="conformal_distribution",  # either "conformal_distribution" or "conformal_error"
        # later I will add the quantile regression too
        level=None,  # The level -> needs to be provided in a list
        pi_windows=None,  # How many windows to get to calculate conformal scores
        user_provided_adjusted_df=None,  # for cases with multiple items, estiamting the adjusted df takes a whille
        # So we estimate it once and provide it to the user!
        user_provided_test_adjusted_df=None,
        future_covars=None,  # If we have covariates for inference
        fill_date_range=True,  # If we fill all dates for all values!
    ):


        logger.info("Starting GlobalForecaster fit method")
        logger.info(f"Memory usage before fit: {psutil.virtual_memory().percent}%")
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Unique IDs: {df['unique_id'].nunique()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")


        self.static_features = static_features
        self.merge_on = merge_on
        self.dropna = dropna
        self.pi_level = level
        self.calculate_intervals = calculate_intervals
        self.filter_zero_lags = filter_zero_lags
        if future_covars is not None:
            future_covars = future_covars.rename(columns={"date": "ds"})
        self.future_covars = future_covars
        self.fill_date_range = fill_date_range

        if user_provided_adjusted_df is None:
            logger.info("Processing and adjusting input data")
            df = statsforecast_forecast_format(df)
            if filter_missing is None:
                filter_missing = max(self.lags)
            self.adjusted_df = filter_items_by_observation(df, filter_missing)
            logger.info(f"Adjusted data shape: {self.adjusted_df.shape}")
        else:
            self.adjusted_df = user_provided_adjusted_df
            logger.info("Using user-provided adjusted data")

        logger.info(f"Memory usage after data processing: {psutil.virtual_memory().percent}%")
        # Calling fit
        if not self.hyperparams_cv:
            self._fit(
                self.adjusted_df,
            filter_zero_lags,
            future_covars=future_covars,
        )

    def predict(
        self,
        h,
        holdout=True,
        cv=1,
        step_size=1,
        refit=True,
        # For in_sample
        in_sample=False,
        cv_heuristic=True,  # A heuristic to drop early values on the cv to reduce computation times
        # Might create problems in some cases, use with caution
        cv_heuristic_value=0.8,  # The % of the values to drop
        # Keep in mind the heuristic works with refit = False only
        calculate_intervals=False,
        pi_windows=None,
        level=None,
        interval_method="empirical",  # supports empirical and conformal
        skipzeros=False,  # skip zeros on the calculation of the quantiles!
        error_type="normal",  # either absolute or normal
        on=[
            "unique_id",
            "fh",
        ],  # where to calculate quantiles from. Across each ts or across ts & horizon
        future_covars=None,
        future_covar_df = None,
    ):

        if not holdout and cv > 1:
            raise ValueError("Cannot do cross validation without holdout.")

        if calculate_intervals and (level is None or pi_windows is None):
            raise ValueError("Provide level and pi_windows to calculate intervals.")

        # Add to the object
        self.cv = cv
        self.h = h
        self.holdout = holdout
        self.step_size = step_size
        self.refit = refit

        # I got rid of these on my new version and are moved to .fit
        # self.pred_df = fill_gaps(self.fit_df, freq=self.freq).fillna(0).drop_duplicates()
        # Fit the model
        # self.forecaster.fit(
        #    self.pred_df,
        #    fitted=self.fitted,
        #    static_features=self.static_features,
        #    dropna=self.dropna,
        #    prediction_intervals=self.pi_estimator,
        # )

        # if cv_heuristic:
        # The heuristic to drop values before the test set!
        #        if refit == True:
        #            raise ValueError("The heuristic works with refit = False only")

        # Drop 80% of the dates!
        #        temp_dates = sorted(self.test_adjusted_df["ds"].unique())
        #        split_date = temp_dates[int(len(temp_dates) * cv_heuristic_value)]

        #        self.test_adjusted_df = self.test_adjusted_df[
        #            self.test_adjusted_df["ds"] > split_date
        #        ]

        if self.holdout:
            # Here we play with cross-validation
            # Remember -> PIs do not work here!

            # If we have distributed -> to be worked on!

            """
            # Cross-validation
            y_pred = self.forecaster.cross_validation(
                #self.adjusted_df,
                self.test_adjusted_df,
                h=self.h,
                n_windows=self.cv,
                step_size=self.step_size,
                refit=self.refit,
                )
            """

            y_pred = self.cross_validation_custom(
                df=self.adjusted_df,
                h=self.h,
                cv=self.cv,
                step_size=self.step_size,
                refit=self.refit,
                future_covars=future_covars,
            )

            self.temp = y_pred

            # Convert to the right format
            # Rename
            y_pred = y_pred.rename(columns={"ds": "date", "y": "True"})

            # Melt -> Deals with cases with multiple models
            y_pred = pd.melt(
                y_pred,
                id_vars=["unique_id", "date", "cutoff", "True"],
                var_name="Model",
                value_name="y",
            )

            # Add the forecast horizon and cross-validation
            y_pred = add_fh_cv(y_pred, self.holdout)

        else:


            # checking for future covars
            #if self.future_covars is not None:
            #    self.adjusted_df = pd.merge(
            #        self.future_covars,
            #   self.adjusted_df,
            #    how="right",
            #        on=["unique_id", "ds"],
            #        )

            if future_covar_df is not None:
                # filter the adjusted df with the unique ids of the future df
                temp_adjusted_df = self.adjusted_df[self.adjusted_df['unique_id'].isin(future_covar_df['unique_id'].unique())]
            else:
                temp_adjusted_df = self.adjusted_df


            y_pred = self.predict_(h = self.h, static_features = self.static_features, new_df = temp_adjusted_df, X_df = future_covar_df)


            #y_pred = self.forecaster.predict(
            #    h=self.h, level=self.pi_level, new_df=temp_adjusted_df,
            #    X_df=future_covar_df
            #)

            y_pred = y_pred.rename(columns={"ds": "date"})

            # Melt
            y_pred = pd.melt(
                y_pred,
                id_vars=["unique_id", "date"],
                var_name="Model",
                value_name="y",
            )

            # Add the forecast horizon and cross-validation
            y_pred = add_fh_cv(y_pred, self.holdout)
            self.y_pred = y_pred

            if calculate_intervals:

                # Cross-validation for pis
                validation_errors = self.cross_validation_custom(
                    # self.adjusted_df,
                    df=self.adjusted_df,
                    h=self.h,
                    cv=pi_windows,  # The number of windows
                    step_size=self.step_size,
                    refit=True,
                )

                self.temp = validation_errors
                # Convert to the right format
                # Rename
                validation_errors = validation_errors.rename(
                    columns={"ds": "date", "y": "True"}
                )

                # Melt -> Deals with cases with multiple models
                validation_errors = pd.melt(
                    validation_errors,
                    id_vars=["unique_id", "date", "cutoff", "True"],
                    var_name="Model",
                    value_name="y",
                )

                validation_errors = add_fh_cv(validation_errors, True)

                # Estimate the quantiles
                quantiles = calculate_empirical_errors(
                    validation_errors,
                    level,
                    method=interval_method,
                    skipzeros=skipzeros,
                    on=on,
                    error_type=error_type,
                )

                # Fix an issue with duplicates columns
                # quantiles = quantiles.loc[:,~quantiles.columns.duplicated()]

                self.quantiles = quantiles

                # Add the intervals
                y_pred = estimate_intervals(y_pred, quantiles, level, on=on)

        # Add to the object
        self.forecast_df = y_pred

        # Then the user can access estimate_in sample and get them
        # To estiamte them check line 539
        if in_sample:
            # Mimicking nixtla code: https://github.com/Nixtla/mlforecast/blob/main/mlforecast/forecast.py
            # line 519
            base = self.feature_df[[self.id_col, self.time_col]]
            fitted_values = self.forecaster._compute_fitted_values(
                base=base,
                X=self.x,
                y=self.y,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
                max_horizon=self.h,
            )
            self.fitted_values = fitted_values

        return y_pred

    def cross_validation_custom(self, df, h, cv, step_size, refit,
                                future_covars=None):

        # Initialize
        results = []
        cv_models = []
        # cv_fitted_values = []

        # First we make the splits!
        splits = ufp.backtest_splits(
            df,  # check if covariates are needed here
            n_windows=cv,
            h=h,
            id_col=self.id_col,
            time_col=self.time_col,
            freq=self.freq,
            step_size=step_size,
            input_size=None,
        )
        self.splits = splits

        # We iterate over the splits
        for i_window, (cutoffs, to_forecast_cv, valid) in enumerate(splits):

            should_fit = i_window == 0 or (refit > 0 and i_window % refit == 0)
            # If we have to refit on every split
            if should_fit:
                self._fit(
                to_forecast_cv,
                self.filter_zero_lags,
                #future_covars=self.future_covars,
                )

            # Adding the models
            cv_models.append(self.forecaster.models_)

            # Prepare and add static covariates here
            # Update. This might not be needed as I split on test_adjusted_df

            # to_forecast_cv = fill_date_gaps_with_covariates(
            #            train,
            #            static_covariates=self.static_features,
            #            merge_on=self.merge_on,
            #            # fill_value_covars=fill_value_covars,
            #            fill_value_covars=None
            #        )

            # I might have to add the heuristic here too though!

            # Add future covariates
            if future_covars is not None:
                future_covar_cols = future_covars + [self.id_col, self.time_col]
                future_covars_df = valid[future_covar_cols]
                self.future_covars_test = future_covars_df
                #to_forecast_cv = to_forecast_cv.drop(columns=future_covars)
            else:
                future_covars_df = None
            
            # Predict
            self.to_forecast_cv = to_forecast_cv
            self.cutoff = cutoffs
            self.valid_c = valid
            # Here I just pass valid with the future covar columns
            #y_pred = self.forecaster.predict(h=h, new_df=to_forecast_cv, X_df=future_covars_df)
            y_pred = self.predict_(h = self.h, static_features = self.static_features,
                                   new_df = to_forecast_cv, X_df = future_covars_df)

            # add cutoffs
            y_pred = ufp.join(y_pred, cutoffs, on=self.id_col, how="left")
            # Merge with the true values
            result = ufp.join(
                valid[[self.id_col, self.time_col, self.target_col]],
                y_pred,
                on=[self.id_col, self.time_col],
            )
            self.b = y_pred
            # Sort indices
            sort_idxs = ufp.maybe_compute_sort_indices(
                result, self.id_col, self.time_col
            )
            self.c = sort_idxs
            self.ww = valid

            if sort_idxs is not None:
                result = ufp.take_rows(result, sort_idxs)
            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency set on the MLForecast constructor matches your series' "
                    "and that there aren't any missing periods."
                )

            # Add to the results of all cvs
            results.append(result)

        # Post process

        self.forecaster.cv_models = cv_models
        del self.forecaster.models_

        out = ufp.vertical_concat(results, match_categories=False)
        out = ufp.drop_index_if_pandas(out)

        first_out_cols = [self.id_col, self.time_col, "cutoff", self.target_col]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]

        return out[first_out_cols + remaining_cols]

    #def save(self, path):
        #"""
        #Saves the model and all the dataframes needed
        #"""

        # Save the fitted model
    #    with fsspec.open(f"{path}/model.pkl", "wb") as f:
            #cloudpickle.dump(self.forecaster.models_, f)

        # Save the ts object
    #    self.forecaster.ts.save(f"{path}/ts.pkl")
        # Save the feature_df
    #    self.feature_df.to_csv(f"{path}/feature_df.csv", index=False)
        # Save the adjusted df
    #    self.adjusted_df.to_csv(f"{path}/adjusted_df.csv", index=False)
        # Save the test_adjusted_df if we have one
    #    self.test_adjusted_df.to_csv(f"{path}/test_adjusted_df.csv", index=False)

        # Add all configurations into a dictionary
        #configurations = {
        #    "lags": self.lags,
        #    "freq": self.freq,
        #    "seasonal_length": self.seasonal_length,
        #    "lag_transforms": self.lag_transforms,
        #    "transformations": self.transformations,
        #    "date_features": self.date_features,
            #"model_names": self.model_names,
        #    "target_col": self.target_col,
        #    "id_col": self.id_col,
        #    "time_col": self.time_col,
        #    "pi_level": None,
        #}
        # Save the configurations
        #with fsspec.open(f"{path}/configurations.pkl", "wb") as f:
        #    cloudpickle.dump(configurations, f)

    #def load(self, path, test=False):
        #""" "
        #Load the fitted model and the dataframes to accompany it
        #"""
        # Load the model and the ts
        #with fsspec.open(f"{path}/model.pkl", "rb") as f:
        #    models = cloudpickle.load(f)

        #ts = TimeSeries.load(f"{path}/ts.pkl")

        # Load the accompaning dfs
        #feature_df = pd.read_csv(f"{path}/feature_df.csv")
        #adjusted_df = pd.read_csv(f"{path}/adjusted_df.csv")
        #test_adjusted_df = pd.read_csv(f"{path}/test_adjusted_df.csv")

        # Load the configurations
        #with fsspec.open(f"{path}/configurations.pkl", "rb") as f:
        #    configurations = cloudpickle.load(f)
        # Define a new Forecaster
        #forecaster = GlobalForecaster(
        #    models=models,
        #    lags=[i for i in range(1, max(configurations["lags"]) + 1)],
        #    freq=configurations["freq"],
        #    transformations=configurations["transformations"],
        #    date_features=configurations["date_features"],
        #)

        # Assing values
        #forecaster.forecaster.models_ = models
        #forecaster.feature_df = feature_df
        #forecaster.adjusted_df = adjusted_df
        #forecaster.test_adjusted_df = test_adjusted_df
        #forecaster.test_adjusted_df["ds"] = pd.to_datetime(
         #   forecaster.test_adjusted_df["ds"]
        #)
        #forecaster.calculate_intervals = False
        #forecaster.forecaster.ts = ts
        #forecaster.pi_level = None

        #if test:
            # Load the adjusted df
            #test_adjusted_df = pd.read_csv(f"{path}/test_adjusted_df.csv")
            #forecaster.test_adjusted_df = test_adjusted_df

        #return forecaster


    def predict_(
        self,
        h: int,
        static_features = None,
        new_df: pd.DataFrame = None,
        X_df: pd.DataFrame = None,
    ):
        """Compute the predictions for the next `h` steps.

        Parameters
        ----------
        h : int
            Number of periods to predict.
        new_df : pandas or polars DataFrame, optional (default=None)
            Series data of new observations for which forecasts are to be generated.
                This dataframe should have the same structure as the one used to fit the model, including any features and time series data.
                If `new_df` is not None, the method will generate forecasts for the new observations.
        X_df : pandas or polars DataFrame, optional (default=None)
            Dataframe with the future exogenous features. Should have the id column and the time column.
        ids : list of str, optional (default=None)
            List with subset of ids seen during training for which the forecasts should be computed.

        Returns
        -------
        result : pandas or polars DataFrame
            Predictions for each serie and timestep, with one column per model.
        """

        if new_df is not None:
            new_ts = TimeSeries(
                freq=self.forecaster.ts.freq,
                lags=self.forecaster.ts.lags,
                lag_transforms=self.forecaster.ts.lag_transforms,
                date_features=self.forecaster.ts.date_features,
                num_threads=self.forecaster.ts.num_threads,
                target_transforms=self.forecaster.ts.target_transforms,
                lag_transforms_namer=self.forecaster.ts.lag_transforms_namer,
            )
            new_ts._fit(
                new_df,
                id_col=self.forecaster.ts.id_col,
                time_col=self.forecaster.ts.time_col,
                target_col=self.forecaster.ts.target_col,
                static_features=static_features,
            )
            #core_tfms = new_ts._get_core_lag_tfms()
            core_tfms = _get_core_lag_tfms(new_ts)
            if core_tfms:
                # populate the stats needed for the updates
                new_ts._compute_transforms(core_tfms, updates_only=False)
            new_ts.max_horizon = self.forecaster.ts.max_horizon
            new_ts.as_numpy = self.forecaster.ts.as_numpy
            ts = new_ts
        else:
            ts = self.forecaster.ts

        forecasts = ts.predict(
            models=self.forecaster.models_,
            horizon=h,
            X_df=X_df,
        )
        return forecasts


def _get_core_lag_tfms(ts):
        return {
            k: v for k, v in ts.transforms.items() if isinstance(v, BaseLagTransform)
        }
def split_X_y(df, target_col, id_col, time_col):

    X = df.drop(columns=[target_col, id_col, time_col])
    y = df[target_col].values

    return X, y


def extract_prediction_intervals(y_pred, levels, model_names):

    # Initialize a dataframe
    out_df = pd.DataFrame()

    # Iterate over the models
    for model in model_names:

        # take the prediction
        y_col = y_pred[model]
        # initiate a temporary dataframe
        temp_df = y_pred[["unique_id", "date"]].copy()
        # Itterate over the level
        for level in levels:
            lo_col = y_pred[
                f"{model}-lo-{level}"
            ]  # Adjust the string format if necessary
            hi_col = y_pred[f"{model}-hi-{level}"]

            # Combine into a temporary dataframe

            temp_df["model"] = model
            temp_df["y"] = y_col
            temp_df[f"Lower-{level}"] = lo_col
            temp_df[f"Upper-{level}"] = hi_col

        # Append to the new dataframe
        out_df = pd.concat([out_df, temp_df], axis=0)

    return out_df


def merge_future_covars(feature_df, covariates_df, merge_on):
    """
    Merge the future covariates with the feature_df


    Currently only supports merge on a single column
    Will need to extend this to include more than one!
    """

    # Validate that the column names are identical
    if "date" in covariates_df.columns:
        covariates_df = covariates_df.rename(columns={"date": "ds"})

    if merge_on == "date":
        merge_on = "ds"

    # Merge
    feature_df = feature_df.merge(covariates_df, how="left", on=merge_on)

    return feature_df


def expand_lag_transforms(lag_transforms):
    """
    Expands lag_transforms keys that are tuples into individual keys and keeps single integer keys as is.

    Args:
        lag_transforms (dict):
            Original dictionary of lag transforms where keys can be integers or tuples of integers.

    Returns:
        dict:
            Modified dictionary with individual keys for each lag.


    Example:
        Input dictionary:
        {
            (1, 2, 3, 4): RollingMean(window_size=7),
            2: RollingMean(window_size=14),
            (1, 2): SeasonalRollingMean(season_length=7, window_size=3),
        }

        Output dictionary:
        {
            1: [RollingMean(window_size=7), SeasonalRollingMean(season_length=7, window_size=3)],
            2: [RollingMean(window_size=7), RollingMean(window_size=14),
                SeasonalRollingMean(season_length=7, window_size=3)],
            3: [RollingMean(window_size=7)],
            4: [RollingMean(window_size=7)]
        }

    """
    expanded_transforms = {}
    for key, transforms in lag_transforms.items():
        if isinstance(key, tuple):  # Check if the key is a tuple and expand
            for lag in key:
                # Ensure we do not overwrite existing keys with the same transformations
                # but instead append if multiple transformations for the same lag
                if (
                    lag in expanded_transforms
                    and transforms not in expanded_transforms[lag]
                ):
                    expanded_transforms[lag].append(transforms)
                else:
                    expanded_transforms[lag] = [transforms]
        else:  # Handle single integer keys correctly
            if (
                key in expanded_transforms
                and transforms not in expanded_transforms[key]
            ):
                expanded_transforms[key].append(transforms)
            else:
                expanded_transforms[key] = [transforms]
    return expanded_transforms


def add_fh_cv(forecast_df, holdout):
    """
    Adds the forecasting horizon and cross-validation information to the forecast results.
    Updated from the Stats version due to different cutoffs per time series.

    Args:
        forecast_df (pd.DataFrame):
            The df containing the forecasted results.
        holdout (bool):
            Whether the forecast is a holdout forecast.

    Returns:
        pd.DataFrame:
            The df containing the forecasted results with the forecasting horizon and cross-validation information.
    """

    # Take all the unique_ids
    ids = forecast_df["unique_id"].unique()

    # Initialize a new dataframe
    out_dataframe = pd.DataFrame()

    # add the number of cv and fh
    if holdout:

        for id in ids:
            # take the unique_id
            temp_df = forecast_df[forecast_df["unique_id"] == id]

            temp_df.loc[:, "cv"] = (
                temp_df.loc[:, "cutoff"].astype("category").cat.codes + 1
            )
            temp_df.loc[:, "fh"] = (
                temp_df.loc[:, "date"] - temp_df.loc[:, "cutoff"]
            ).dt.days

            # concat
            out_dataframe = pd.concat([out_dataframe, temp_df])

    else:
        for id in ids:

            # take the unique_id
            temp_df = forecast_df[forecast_df["unique_id"] == id]
            # Encode fh
            temp_df.loc[:, "fh"] = (
                temp_df.loc[:, "date"].astype("category").cat.codes + 1
            )

            # concat
            out_dataframe = pd.concat([out_dataframe, temp_df])

        # also add the cv
        out_dataframe["cv"] = None

    return out_dataframe


def filter_items_by_observation(
    df, min_non_zero_observations, value_col="y", id_col="unique_id"
):
    """
    Filter out items that have less than the provided number of non-zero observations.

    """
    # Count the number of non-zero observations for each unique_id
    counts = df.groupby("unique_id").count().sort_values("y")["y"]

    # Filter out unique_ids that don't meet the min_non_zero_observations threshold
    filtered_ids = counts[counts >= min_non_zero_observations].index

    # Return a DataFrame containing only the filtered unique_ids
    return df[df[id_col].isin(filtered_ids)]


def fill_date_gaps_with_covariates(
    df,
    static_covariates=None,
    merge_on=None,
    fill_value=0,
    fill_value_covars=None,
    fill_date_range=True,
):
    """
    Fill the gaps between observations in the dataframe for each unique_id,
    merging static covariates based on date.

    If we have covariates (either for the dates or the item) i am keeping them seperate and then remerge
    This might need some attention on the future

    This funcion will need further processing!

    static_covariates : list of str
        The static covariates to merge back into the dataframe.
        Since I am adding extra rows I dont have information on this. Thus it needs to be given.
    merge_on : list of str or str
        The column to perform merge of the covariates on.
        If covariates are on the item description then unique_id. If they are on date, then date.
        For combinations of date and item covariates, then provide a list with the same order as the static_covariates
    fill_value : int, float, str
        The value to use for filling missing values in the y column
    fill_value_covars : int, float, str
        The value to use for filling missing values in the covariates columns
        We might have a problem here if I have multiple covariates. Will investigate!
    fill_date_range: bool, default = True
        Boolean if we fit the full date range!
    """

    # Initializing some values
    date_col = "ds"
    id_col = "unique_id"
    value_col = "y"

    # Make some assertions
    if static_covariates is not None:
        # Convert to the right format
        static_covariates, merge_on = check_covars(static_covariates, merge_on)
        # Extract static covariates into a separate DataFrame if they are provided
        cols = list(set(merge_on)) + static_covariates
        covariate_df = df[cols].drop_duplicates()

    # Start with an empty DataFrame to hold the results
    result_df = pd.DataFrame()

    # Process each unique_id separately
    for uid in df[id_col].unique():
        # Filter the dataframe for the current unique_id
        uid_df = df[df[id_col] == uid]

        if static_covariates is not None:
            uid_df = uid_df.drop(static_covariates, axis=1)

        if fill_date_range:
            # Create a date range from min to max date for the current unique_id
            date_range = pd.date_range(
                start=uid_df[date_col].min(), end=uid_df[date_col].max()
            )

            # Reindex the uid_df to have a row for each date in the date range
            uid_df = (
                uid_df.set_index(date_col)
                .reindex(date_range)
                .rename_axis(date_col)
                .reset_index()
            )

            # Fill missing values in the value column with the specified fill_value
            uid_df[value_col] = uid_df[value_col].fillna(fill_value)

            # Fill the id_col with the unique_id as this is not a static covariate
            uid_df[id_col] = uid

        # Merge the static covariates back into uid_df
        if static_covariates:

            # for every pair covariate, merge on
            for covar, merge in zip(static_covariates, merge_on):

                # Filter the covariate df to keep only the pair of the itteration
                temp_covariate_df_ = covariate_df[[merge, covar]].drop_duplicates()

                # MErge
                uid_df = uid_df.merge(
                    temp_covariate_df_, how="left", on=merge
                ).drop_duplicates()

            # uid_df = uid_df.merge(covariate_df, how='left', on = merge_on).drop_duplicates()

        # Append the uid_df to the result_df
        result_df = pd.concat([result_df, uid_df], ignore_index=True)

    if fill_value_covars is not None:
        result_df[static_covariates] = result_df[static_covariates].fillna(
            fill_value_covars
        )

    # Ensure no nans -> drop them
    result_df = result_df.dropna()

    # droping duplicates is a brute force strategy. Will have to fix it
    return result_df.drop_duplicates(subset=[id_col, date_col])


def filter_identical_lag_rows(df, target_col="y"):
    """
    Filters out rows where all lag features and the target column have the same value.

    :param df: DataFrame with the features and target.
    :param target_col: Name of the target column.
    :return: DataFrame filtered to exclude rows with identical lag and target values.
    """
    # Find all columns that contain the word "lag"
    lag_cols = [col for col in df.columns if "lag" in col]

    # Check if all lag values and the target value are equal to 0 for each row
    identical_values_mask = df[lag_cols + [target_col]].apply(
        lambda row: (row.nunique() == 1) & (row.iloc[0] == 0), axis=1
    )
    # Invert the mask to filter out rows where the lag and target values are identical
    filtered_df = df[~identical_values_mask]

    return filtered_df


def calculate_empirical_errors(
    df,
    level,
    method="empirical",
    skipzeros=True,
    error_type="absolute",
    on=["unique_id", "fh"],
):

    if error_type == "absolute":
        # Estimate the absolute errors
        df["error"] = abs(df["y"] - df["True"])
    elif error_type == "normal":
        df["error"] = df["True"] - df["y"]
    elif error_type == "positive":
        df["error"] = np.maximum(0, df["True"] - df["y"])
    else:
        raise ValueError("Error type should be either absolute or normal")

    # Check if level is of instance list
    if not isinstance(level, list):
        level = [level]

    # print(level)
    if level[0] >= 1:

        level = [x / 100 for x in level]

    if method == "empirical":
        upper_bound = level
        # skipzeros = True
        # df['error'] = abs(df['error'])

    elif method == "conformal":
        skipzeros = False

        # NOTE: Made the change bellow!!!
        # Verify the last week that it is correct!
        # alpha = [100 - temp_level for temp_level in level]
        # alpha = level
        # upper_bound = [1 - a / 200 for a in alpha]
        # upper_bound = [1 - a / 2 for a in alpha]

        upper_bound = level

    else:
        raise ValueError("Method should be either empirical or conformal")

    all_quants_df = pd.DataFrame()

    for q, m in zip(upper_bound, level):
        # Calculate the quantiles for each group from the filtered DataFrame
        if skipzeros:
            non_zero = df[df["error"] != 0]
            quantile_df = (
                non_zero.groupby(on)["error"]
                .quantile(q, numeric_only=True)
                .to_frame()
                .reset_index()
            )
            extended_on = on + ["error"]
            zero = df[df["error"] == 0][extended_on]
            # Groupby on unique_id and fh and repeat zeroes
            zero = zero.groupby(on)["error"].mean().reset_index()

            # Concatenate the two dataframes
            quantile_df = pd.concat([quantile_df, zero], axis=0)

            # Drop duplicates on unique_id, fh, keep first as its with the non zero
            quantile_df = quantile_df.drop_duplicates(subset=on, keep="first")

        else:
            quantile_df = df.groupby(on)["error"].quantile(q).to_frame().reset_index()

        quantile_df = quantile_df.set_index(on)
        # Rename the columns to reflect the actual quantiles

        level_percentage = int(
            round(m * 100, 2)
        )  # Convert level to a percentage for labeling
        # print(m,level_percentage )
        quantile_df = quantile_df.rename(
            columns={"error": f"upper_{level_percentage}%"}
        )

        # Merge
        all_quants_df = pd.concat([all_quants_df, quantile_df], axis=1)

    # Melt
    # quantile_df = quantile_df.reset_index().melt(id_vars = ['unique_id'],  va)

    return all_quants_df.reset_index()


def estimate_intervals(
    predictions,
    quantiles,
    level,
    on=["unique_id", "fh"],
):

    if not isinstance(level, list):
        level = [level]

    if level[0] >= 1:
        # NOTE: MADE THE CHANGE BELLOW!
        # level = [100 - temp_level for temp_level in level]
        level = [temp_level / 100 for temp_level in level]
    # Add the quants to test
    predictions = pd.merge(predictions, quantiles, on=on, how="left")
    # print(predictions.columns)
    # Add the interval
    for quant in level:
        # print(quant)
        val = int(round(quant * 100, 2))  # could not avoid the round,int
        col = f"upper_{val}%"
        predictions[f"up_{val}"] = predictions["y"] + predictions[col]

        # Keep the col as it is if its over 0 else convert to 0
        predictions[f"up_{val}"] = predictions[f"up_{val}"].apply(
            lambda x: x if x > 0 else 0
        )
        predictions = predictions.drop(col, axis=1)

    return predictions


class StatisticalForecaster(object):
    """
    A class for time series forecasting using statistical methods.

    Methods:
        __init__(models, freq, n_jobs=-1, warning=False, seasonal_length=None)
            Initialize the StatisticalForecaster object.
        fit(df, format="pivoted", fallback=True, verbose=False)
            Fit the model to given the data.
        predict(h, cv=1, step_size=1, refit=True, holdout=True)
            Generates predictions using the statistical forecaster.
        calculate_residuals()
            Calculates the residuals of the model.
        residuals_diagnosis(model = 'ETS', type = 'random', n = 3)
            Plots diagnosis for the residuals
        add_fh_cv()
            Adds the forecast horizon and cross-validation to the predictions.

    Args:
        models: list
            A list of models to fit.
        freq: str
            The frequency of the data, e.g. 'D' for daily or 'M' for monthly.
        n_jobs: int, default=-1
            The number of jobs to run in parallel for the fitting process.
        warning: bool, default=False
            Whether to show warnings or not.
        seasonal_length: int, default=None
            The length of the seasonal pattern.
            If not given, it is inferred from the frequency.
        df: pd.DataFrame
            The input data.
        format: str, default='pivoted'
            The format of the input data.
            Can be 'pivoted' or 'transactional'.
        fallback: bool, default=True
            Whether to fallback to the default model if the model fails to fit.
            Default selection is Naive
        verbose: bool, default=False
            Whether to show the progress of the model fitting.
        h: int
            The forecast horizon.
        cv: int, default=1
            The number of cross-validations to perform.
        step_size: int, default=1
            The step size for the cross-validation.
        refit: bool, default=True
            Whether to refit the model on the entire data after cross-validation.
        holdout: bool, default=True
            Whether to hold out the last observation for cross-validation.
        model: str, default='ETS'
            The model to plot the residuals for.
        type: str, default='random'
            The type of residuals to plot. Can be 'random', aggregate, individual.
        n: int, default=3
            The number of residuals to plot.

    Examples:
        # Create the forecaster
        >>> models = ["ETS"]
        >>> freq = "M"
        >>> n_jobs = -1
        >>> forecaster = StatisticalForecaster(models, freq, n_jobs)

        # Fit the forecaster
        >>> df = pd.read_csv("data.csv")
        >>> forecaster.fit(df, format="pivoted")

        # Generate predictions
        >>> h = 12
        >>> cv = 3
        >>> holdout = True
        >>> predictions = forecaster.predict(h, cv, holdout)


    """

    def __init__(
        self,
        models,
        freq,
        n_jobs=1,
        warning=False,
        seasonal_length=None,
        distributed=False,
        n_partitions=None,
        window_size=None,
        seasonal_window_size=None,
    ):
        """
        Initialize the StatisticalForecaster object.

        Args:
            models: list
                A list of models to fit. Currently only ETS is implemented.
            freq: str
                The frequency of the data, e.g. 'D' for daily or 'M' for monthly.
            n_jobs: int, default=1
                The number of jobs to run in parallel for the fitting process.
            warning: bool, default=False
                Whether to show warnings or not.
            seasonal_length: int, default=None
                The length of the seasonal pattern.
                If None, the seasonal length is inferred from the frequency.
                On frequencies with multiple seasonal patterns, the first seasonal pattern is used.

        """
        self.freq = freq
        if seasonal_length is not None:
            self.seasonal_length = seasonal_length
        else:
            self.seasonal_length = get_numeric_frequency(freq)
            # Check if it returns multiple seasonal lengths
            if isinstance(self.seasonal_length, list):
                # take the first
                self.seasonal_length = self.seasonal_length[0]
        self.n_jobs = n_jobs

        # Set the warnings
        if not warning:
            warnings.filterwarnings("ignore")

        # Add the models and their names
        models_to_fit = []
        model_names = []

        # Converts models to statsforecast objects
        models_to_fit, model_names = model_selection(
            models, self.seasonal_length, window_size, seasonal_window_size
        )

        self.fitted_models = models_to_fit
        self.model_names = model_names

        self.distributed = distributed
        self.n_partitions = n_partitions
        # Initiate FugueBackend with DaskExecutionEngine if distributed is True
        #if self.distributed:
            #dask_client = Client()
            #engine = DaskExecutionEngine(dask_client=dask_client)  # noqaf841

    def fit(self, df, format="pivoted", fallback=True, verbose=False):
        """
        Fit the model to given the data.

        Args:
            df : pd.DataFrame
                The input data.
            format : str, default='pivoted'
                The format of the input data. Can be 'pivoted' or 'transactional'.
            fallback : bool, default=True
                Whether to fallback to the default model if the model fails to fit.
                Default selection is Naive
            verbose : bool, default=False
                Whether to show the progress of the model fitting.
        Raises:
            ValueError : If the format is not 'pivoted' or 'transactional'.

        """

        if format == "pivoted":
            fc_df = transaction_df(df, drop_zeros=False)
        elif format == "transactional":
            fc_df = df.copy()
        else:
            raise ValueError(
                "Provide the dataframe either in pivoted or transactional format."
            )

        # convert to the right format for forecasting
        fc_df = statsforecast_forecast_format(fc_df)

        # Define the StatsForecaster
        if fallback:
            self.forecaster = StatsForecast(
                df=fc_df,
                models=self.fitted_models,
                freq=self.freq,
                n_jobs=self.n_jobs,
                fallback_model=Naive(),
                verbose=verbose,
            )
        else:
            self.forecaster = StatsForecast(
                df=fc_df,
                models=self.fitted_models,
                freq=self.freq,
                n_jobs=self.n_jobs,
                verbose=verbose,
            )

        # Check if we have distributed training
        #if self.distributed:
            # Convert the df to a dask dataframe
        #    fc_df = dd.from_pandas(fc_df, npartitions=self.n_partitions)

        # Add to the object
        self.fc_df = fc_df
        # Add statsforecast names to the object
        self.statsforecast_names = [
            str(model_name) for model_name in self.forecaster.models
        ]

    def predict(
        self,
        h,
        cv=1,
        step_size=1,
        refit=True,
        holdout=True,
        calculate_pis=False,
        level=None,
        invert_intervals=False,
        non_negative=False,
    ):
        """
        Generates predictions using the statistical forecaster.

        Args:
            h : int
                The forecast horizon (i.e., how many time periods to forecast into the future).
            cv : int, optional (default=1)
                The number of cross-validation folds to use. If set to 1, no cross-validation is performed.
            step_size : int, optional (default=1)
                The step size to use for cross-validation. If set to 1, the cross-validation folds are non-overlapping
            refit : bool, optional (default=True)
                Weather to refit the model at each cross-validation. Avoid for big datasets.
            holdout : bool, optional (default=True)
                If True, a holdout set is used for testing the model. If False, the model is fit on the entire data.

        Raises:
            ValueError : If cv > 1 and holdout is False.

        Returns:
            pandas.DataFrame
            The forecasted values, along with the true values (if holdout=True).

        """
        if not holdout and cv > 1:
            raise ValueError("Cannot do cross validation without holdout.")

        if holdout and cv is None:
            cv = 1

        if calculate_pis and level is None:
            raise ValueError("Level must be provided for prediction intervals")

        if level is not None and calculate_pis is False:
            calculate_pis = True

        if level is not None:
            # convert to list if we have a single number
            if not isinstance(level, list):
                level = [level]
            # Convert to %
            if level[0] < 1:
                level = [int(i * 100) for i in level]

        # Add to the object
        self.cv = cv
        self.h = h
        self.holdout = holdout
        self.step_size = step_size
        self.refit = refit
        self.level = level
        self.calculate_pis = calculate_pis

        if holdout:
            if self.distributed:
                y_pred = self.forecaster.cross_validation(
                    df=self.fc_df,
                    h=self.h,
                    step_size=self.step_size,
                    n_windows=self.cv,
                    refit=self.refit,
                    level=self.level,
                ).compute()  # add the compute here
            else:
                y_pred = self.forecaster.cross_validation(
                    df=self.fc_df,
                    h=self.h,
                    step_size=self.step_size,
                    n_windows=self.cv,
                    refit=self.refit,
                    level=self.level,
                )

            # edit the format
            # Reset index and rename

            y_pred = y_pred.reset_index().rename(columns={"ds": "date", "y": "True"})

            if calculate_pis:
                # Melt the intervals!
                y_pred = self.melt_intervals(
                    y_pred, invert_intervals=invert_intervals, non_negative=non_negative
                )
            else:
                y_pred = pd.melt(
                    y_pred,
                    id_vars=["unique_id", "date", "cutoff", "True"],
                    var_name="Model",
                    value_name="y",
                )

        else:
            # We just forecast
            # If we have distributed
            if self.distributed:
                y_pred = self.forecaster.forecast(
                    df=self.fc_df,
                    h=self.h,
                    level=self.level,
                ).compute()  # add the compute here
            else:
                y_pred = self.forecaster.forecast(
                    df=self.fc_df,
                    h=self.h,
                    level=self.level,
                )

            # edit the format
            # Reset index and rename
            y_pred = y_pred.reset_index().rename(columns={"ds": "date"})

            if calculate_pis:
                # Melt the intervals!
                y_pred = self.melt_intervals(
                    y_pred,
                    holdout=False,
                    invert_intervals=invert_intervals,
                    non_negative=non_negative,
                )

            else:
                y_pred = pd.melt(
                    y_pred,
                    id_vars=["unique_id", "date"],
                    var_name="Model",
                    value_name="y",
                )

        # Add to the object
        self.forecast_df = y_pred

        # add the fh and cv
        self.forecast_df = add_fh_cv(self.forecast_df, self.holdout)

        # Remove the index from the models if there
        self.forecast_df = self.forecast_df[self.forecast_df["Model"] != "index"]

        # return
        return self.forecast_df

    def calculate_residuals(self, type="default"):
        """
        Calculate residuals for all horizons.

        Args:
            type: str, optional (default='default')
                The type of residuals to calculate. Options are 'default' and 'multistep'.

        Returns:
            pandas.DataFrame : The residuals for all models and horizons.

        """

        # Ensure type is either default or multistep
        if type not in ["default", "multistep"]:
            raise ValueError("Type must be either 'default' or 'multistep'.")

        # Uses statsmodels for getting the residuals
        # statsforecast is buggy
        res = self.calculate_residuals_statsmodels(type="default")

        # add the number of cv and fh
        cv_vals = sorted(res["cutoff"].unique())
        cv_dict = dict(zip(cv_vals, np.arange(1, len(cv_vals) + 1)))
        res["cv"] = [cv_dict[date] for date in res["cutoff"].values]

        # add the fh
        fh_vals = np.tile(np.arange(1, self.h + 1), int(len(res) / self.h))
        res["fh"] = 1 if type == "default" else fh_vals

        # add the residuals
        self.residuals = res

        # return
        return self.residuals

    #def calculate_residuals_statsmodels(self, type):
    #    """
    #    Calculates residuals using the statsmodels ETS
    #    It is used as a fallback when statsforecast fails
    #    It fails when len(y) < nparams + 4 where nparams the number of ETS parameters
    #
    #    Args:
    #        type: str, optional (default='default')
    #            The type of residuals to calculate.
    #            Options are 'default' and 'multistep'.
    #
    #    Returns:
    #        pandas.DataFrame : The residuals for all models and horizons.
    #
    #    """
    #
    #    # Initialize simmulation parameters
    #    end_date = self.h + self.cv - 1
    #    fitting_periods = sorted(self.fc_df["ds"].unique())[:-end_date]
    #    total_windows = len(fitting_periods) - self.h + 1
    #
    #    # Pivot the dataframe
    #    temp_df = pd.pivot_table(
    #        self.fc_df, index="unique_id", columns="ds", values="y", aggfunc="first"
    #    )
    #
    #    # Initialize a df
    #    temp_residuals = pd.DataFrame()
    #
    #    # Itterate over each time series
    #    for i, row in temp_df.iterrows():
    #        # Cut row at the end date
    #        row = row[:-end_date]
    #
    #        model = ETSModel(row, seasonal_periods=self.seasonal_length)
    #        fit = model.fit(disp=False)
    #        # initialie a df
    #        in_sample_df = pd.DataFrame()
    #
    #        if type == "multistep":
    #            # Get multi-step in-sample predictions
    #            for i in range(total_windows - 1):
    #                # Run the simulation
    #                in_sample_multistep = fit.simulate(
    #                    nsimulations=self.h, anchor=i, repetitions=1, random_errors=None
    #                ).to_frame()
    #                # add the cutoff
    #                in_sample_multistep["cutoff"] = fitting_periods[i]
    #                # add to the df
    #                in_sample_df = pd.concat(
    #                    [in_sample_df, in_sample_multistep], axis=0
    #                )
    #        else:
    #            # get the fitted values
    #            in_sample_df = fit.fittedvalues.to_frame()
    #
    #        # Edit the format
    #        # add the unique_id
    #        in_sample_df["unique_id"] = row.name
    #        # Add the true values
    #        row = row.to_frame()
    #        row.columns = ["y_true"]
    #        in_sample_df = in_sample_df.merge(row, left_index=True, right_index=True)
    #        # rename
    #        in_sample_df = in_sample_df.rename(
    #            columns={"simulation": "y_pred", 0: "y_pred"}
    #        )
    #        # reset index
    #        in_sample_df = in_sample_df.reset_index(names="date")
    #        # add the cutoff for default tyoe
    #        if type == "default":
    #            in_sample_df["cutoff"] = in_sample_df["date"].shift(1)
    #            # drop the first row
    #            in_sample_df = in_sample_df.dropna()
    #
    #        # add to the df
    #        temp_residuals = pd.concat([temp_residuals, in_sample_df], axis=0)
    #
    #    # add the Model
    #    temp_residuals["Model"] = "AutoETS"
    #
    #    # Calculate the residuals
    #    temp_residuals["residual"] = temp_residuals["y_true"] - temp_residuals["y_pred"]
    #
    #    return temp_residuals

    def melt_intervals(
        self, predictions, holdout=True, invert_intervals=False, non_negative=False
    ):
        """
        Transforms a forecasting DataFrame with intervalsto the universal format

        Args:
            predictions (pd.DataFrame): The DataFrame to transform.
            holdout (bool, optional): Whether the DataFrame contains holdout data. Defaults to True.
            invert_intervals (bool, optional): Whether to invert the intervals. Defaults to False.
            non_negative (bool, optional): Whether to convert negative intervals to zero. Defaults to False.
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """

        if holdout:
            id_vars = ["unique_id", "date", "cutoff", "True"]
            id_vars_pivot = ["unique_id", "date", "cutoff", "True", "Model"]
            id_vars_melt = ["unique_id", "date", "cutoff", "True", "Model", "Value"]
            id_vars_sort = ["unique_id", "date", "cutoff", "True", "Model", "y"]
        else:
            # Drop the True and the cutoff from all lists
            id_vars = ["unique_id", "date"]
            id_vars_pivot = ["unique_id", "date", "Model"]
            id_vars_melt = ["unique_id", "date", "Model", "Value"]
            id_vars_sort = ["unique_id", "date", "Model", "y"]

        # Determine which models have interval data
        models_with_intervals = [
            model
            for model in self.statsforecast_names
            if any(
                f"-{level}" in col
                for level in self.level
                for col in predictions.columns
                if model in col
            )
        ]
        models_without_intervals = [
            model
            for model in self.statsforecast_names
            if model not in models_with_intervals
        ]

        # Prepare value variables for melting, considering models with and without intervals
        value_vars_with_intervals = [
            f"{model}-{suffix}-{level}"
            for model in models_with_intervals
            for suffix in ["lo", "hi"]
            for level in self.level
        ] + models_with_intervals
        value_vars_without_intervals = models_without_intervals
        value_vars = value_vars_with_intervals + value_vars_without_intervals

        # Melt the DataFrame
        melted = pd.melt(
            predictions,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Model_Level",
            value_name="Value",
        )
        melted["Model_Level"] = (
            melted["Model_Level"].str.replace("lo-", "low_").str.replace("hi-", "up_")
        )

        # Split 'Model_Level' into separate 'Model' and 'Level' columns
        melted[["Model", "Level"]] = melted["Model_Level"].str.split("-", expand=True)

        # Separate forecasts from intervals
        forecast_data = melted[melted["Level"].isna()][id_vars_melt]
        forecast_data = forecast_data.rename(columns={"Value": "y"})

        # Process interval data if present
        if models_with_intervals:
            pivot_cols = "Level"
            interval_data = melted.dropna(subset=id_vars_pivot).copy()
            interval_data_wide = interval_data.pivot_table(
                index=id_vars_pivot, columns=pivot_cols, values="Value", aggfunc="first"
            ).reset_index()

            # Merge forecast data with interval data
            final_df = pd.merge(
                forecast_data, interval_data_wide, on=id_vars_pivot, how="left"
            )
        else:
            # No models with intervals, so just use the forecast data
            final_df = forecast_data.copy()
            for level in self.level:
                final_df[f"low_{level}"] = None
                final_df[f"up_{level}"] = None

        # Convert negative intervals to zero if non_negative is True
        if non_negative:
            for level in self.level:
                final_df[f"low_{level}"] = final_df[f"low_{level}"].apply(
                    lambda x: 0 if x < 0 else x
                )
                final_df[f"up_{level}"] = final_df[f"up_{level}"].apply(
                    lambda x: 0 if x < 0 else x
                )

        # Reverse for example low 60 to up 40!
        if invert_intervals:

            final_df = final_df.rename(
                columns={
                    f"low_{level}": f"up_{100 - int(level)}" for level in self.level
                }
            )
            # sort the columns
            interval_columns = sorted(
                [col for col in final_df.columns if col not in id_vars_sort],
                key=lambda x: (x.split("_")[0], int(x.split("_")[1])),
            )
            final_df = final_df[id_vars_sort + interval_columns]

        return final_df


def model_selection(models, seasonal_length, window_size, seasonal_window_size):
    "Takes models in a list of strings and returns a list of Statsforecast objects"

    # Initiate the lists
    # Add the models and their names
    models_to_fit = []
    model_names = []

    # Append to the list
    if "Naive" in models:
        models_to_fit.append(Naive())
        model_names.append("Naive")
    if "SNaive" in models:
        models_to_fit.append(SeasonalNaive(season_length=seasonal_length))
        model_names.append("Seasonal Naive")
    if "ARIMA" in models:
        models_to_fit.append(AutoARIMA(season_length=seasonal_length))
        model_names.append("ARIMA")
    if "ETS" in models:
        models_to_fit.append(AutoETS(season_length=seasonal_length))
        model_names.append("ETS")
    if "CrostonClassic" in models:
        models_to_fit.append(CrostonClassic())
        model_names.append("CrostonClassic")
    if "CrostonOptimized" in models:
        models_to_fit.append(CrostonOptimized())
        model_names.append("CrostonOptimized")
    if "SBA" in models:
        models_to_fit.append(CrostonSBA())
        model_names.append("SBA")
    if "WindowAverage" in models:
        # Assert we have window size
        assert window_size is not None, "Window size must be provided for WindowAverage"
        models_to_fit.append(WindowAverage(window_size=window_size))
        model_names.append("WindowAverage")
    if "SeasonalWindowAverage" in models:
        # Assert we have window size
        assert (
            seasonal_window_size is not None
        ), "Window size must be provided for SeasonalWindowAverage"
        models_to_fit.append(
            SeasonalWindowAverage(
                window_size=seasonal_window_size, season_length=seasonal_length
            )
        )
        model_names.append("SeasonalWindowAverage")

    if "CES" in models:
        models_to_fit.append(AutoCES(season_length=seasonal_length))
        model_names.append("CES")

    if "Theta" in models:
        models_to_fit.append(AutoTheta(season_length=seasonal_length))
        model_names.append("Theta")

    return models_to_fit, model_names
