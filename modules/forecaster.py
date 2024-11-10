from logs.logging_config import setup_logger
from modules.utils.utils import pivoted_df
from modules.utils.forecasting import StatisticalForecaster, GlobalForecaster
import lightgbm as lgb
#import xgboost as xgb
#import catboost as cb
import pandas as pd
from modules.utils.utils import get_numeric_frequency
import psutil

# Set up logger
logger = setup_logger(__name__, 'forecaster.log')

class Forecaster:
    def __init__(
        self,
        model_config: dict,
        forecast_config: dict,
        seasonal_length  = None
    ):
        """
        Initialize the Forecaster with model and forecast configurations.
        """
        logger.info("Initializing Forecaster")
        
        self.fit_benchmarks = model_config["fit_benchmarks"]
        self.holdout = forecast_config["holdout"]

        # Clean and validate the model configuration
        self.model_config = self.check_model_config(model_config)

        # Check and validate the forecast configuration
        self.forecast_config = self.check_forecasting_config(forecast_config)


        if self.fit_benchmarks:
            logger.info("Initializing statistical forecaster for benchmarks")
            self.statistical_forecaster = StatisticalForecaster(
                models=self.model_config["benchmark_models"],
                freq=self.forecast_config["freq"],
                seasonal_length=seasonal_length,
                seasonal_window_size = 4
            )

        logger.info("Initializing global forecaster with configurations: %s", self.model_config)
        self.global_forecaster = GlobalForecaster(
            models=self.model_config["global_models"],
            lags=self.model_config["total_lags"],
            freq=self.forecast_config["freq"],
            transformations=self.model_config["transformations"],
            date_features=self.model_config["seasonal_features"],
            seasonal_length=seasonal_length,
            lag_transforms = self.model_config["lag_transforms"]
        )

        self.predictions = None
        logger.info("Forecaster initialization complete")

    def fit(self, df):
        """
        Fit the model to the data.
        """
        logger.info("Fitting the model to the data")
        logger.info(f"Memory usage before fitting: {psutil.virtual_memory().percent}%")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Unique IDs: {df['unique_id'].nunique()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        if self.fit_benchmarks:
            logger.info("Preparing and fitting statistical benchmarks")
            self.statistical_forecaster.fit(df,  format="transactional")
            logger.info(f"Memory usage after fitting benchmarks: {psutil.virtual_memory().percent}%")

        logger.info("Fitting the global forecaster")
        self.global_forecaster.fit(df[self.forecast_config["selected_features"]], static_features = self.forecast_config["static_features"])
        logger.info(f"Memory usage after fitting global forecaster: {psutil.virtual_memory().percent}%")
        logger.info("Model fitting complete")

    def check_forecasting_config(self, forecast_config):
        """
        Check and validate the forecasting configuration.
        """
        logger.info("Checking forecasting configuration")
        assert "freq" in forecast_config, "Frequency is missing"
        assert "h" in forecast_config, "Number of steps ahead is missing"

        if self.holdout:
            forecast_config.setdefault("cv", 1)
        else:
            forecast_config.setdefault("cv", 0)

        forecast_config.setdefault("n_jobs", 1)

        logger.info("Forecasting configuration checked and validated")
        return forecast_config

    def check_model_config(self, model_config):
        """
        Check and validate the ML configuration.
        """
        logger.info("Checking model configuration")
        assert "lags" in model_config, "Lags are missing"
        model_config["total_lags"] = list(range(1, model_config["lags"] + 1))

        model_config.setdefault("loss", "mse")
        model_config.setdefault("params", {})
        model_config.setdefault("rs", 42)
        model_config.setdefault("model_name", "lgbm")

        assert "model" in model_config, "Model is missing"


        model_config.setdefault("selected_features", ["date", "unique_id", "y"])
        if len(model_config["selected_features"]) > 3:
            # take the extra features as coariatees
            covariates = [ col for col in model_config["selected_features"] if col not in ["date", "unique_id", "y"]]
            model_config["covariates"] = covariates
            model_config['xgb_cat'] = True
        
        else:
            model_config.setdefault("covariates", None)
            model_config['xgb_cat'] = False


        if model_config["model"] == "lgbm":
            logger.info("Initializing LightGBM model")
            forecasting_model = lgb.LGBMRegressor(
                objective=model_config["loss"],
                random_state=model_config["rs"],
                verbose=-1,
                #categorical_feature=model_config["covariates"],
                **model_config["params"]
          )
        #elif model_config["model"] == "xgb":
        #    logger.info("Initializing XGBoost model")
        #    forecasting_model = xgb.XGBRegressor(
        #        objective=model_config["loss"],
        #        random_state=model_config["rs"],
        #        verbose=-1,
        #        enable_categorical=model_config["xgb_cat"],
        #        **model_config["params"]
        #    )
        #elif model_config["model"] == "cb":
        #    logger.info("Initializing CatBoost model")
        #    forecasting_model = cb.CatBoostRegressor(
        #        verbose = 0,
        #        loss_function=model_config["loss"],
        #        random_state=model_config["rs"],
        #        cat_features=model_config["covariates"],
        #        **model_config["params"]
        #    )
        else:
            logger.error(f"Unsupported model: {model_config['model']}")
            raise ValueError("Model currently not supported")

        model_config["global_models"] = {model_config["model_name"]: forecasting_model}

        
        model_config.setdefault("seasonal_features", None)
        model_config.setdefault("transformations", None)
        model_config.setdefault("lag_transforms", None)

        if self.fit_benchmarks:
            assert "benchmark_models" in model_config, "Benchmark models are missing"

        logger.info("Model configuration checked and validated")
        return model_config

    def predict(self, future_df = None):
        """
        Predict the future values.
        """
        logger.info("Starting prediction process")
        logger.info(f"Memory usage before prediction: {psutil.virtual_memory().percent}%")
        try:
            global_predictions = self.global_forecaster.predict(
                h=self.forecast_config["h"],
                cv=self.forecast_config["cv"],
                holdout=self.forecast_config["holdout"],
                future_covar_df = future_df
            )
            logger.info("Global predictions generated successfully")
            logger.info(f"Global predictions shape: {global_predictions.shape}")
        except Exception as e:
            logger.error(f"Error in global forecaster prediction: {str(e)}", exc_info=True)
            raise

        if self.fit_benchmarks:
            logger.info("Generating predictions from benchmark models")
            try:
                benchmark_predictions = self.statistical_forecaster.predict(
                    h=self.forecast_config["h"],
                    cv=self.forecast_config["cv"],
                    holdout=self.holdout,
                )
                global_predictions = pd.concat(
                    [global_predictions, benchmark_predictions], axis=0
                )
                logger.info("Benchmark predictions generated and concatenated successfully")
                logger.info(f"Final predictions shape: {global_predictions.shape}")
            except Exception as e:
                logger.error(f"Error in benchmark prediction: {str(e)}", exc_info=True)
                raise

        self.predictions = global_predictions

        logger.info(f"Memory usage after prediction: {psutil.virtual_memory().percent}%")
        logger.info("Prediction process complete")
        return global_predictions

    
