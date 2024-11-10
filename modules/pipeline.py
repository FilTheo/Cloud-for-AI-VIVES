# Import your modules
from logs.logging_config import setup_logger
from modules.data_preprocessor import DataPreprocessor
from modules.forecaster import Forecaster
from modules.data_extractor import Extractor
from modules.evaluator import ForecastEvaluator
from modules.plots import make_plots
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import re
from datetime import datetime
import warnings
import json
import io
import numpy as np
#from modules.client_preprocess import preprocess_pipeline
import psutil

# Set up logger
logger = setup_logger(__name__, "pipeline.log")

def get_latest_files(config_manager):
    """
    Retrieve the latest files from an S3 bucket.

    Args:
        config_manager (ConfigManager): The configuration manager object.

    Returns:
        tuple: Latest file, latest main file, latest forecast file, latest evaluation file, and latest confirmed file names.
    """
    s3_config = config_manager.s3_config
    bucket_name = s3_config['bucket_name']
    bucket_path = s3_config['data_path']

    logger.info(f"Bucket name: {bucket_name}, Bucket path: {bucket_path}")

    try:
        s3 = boto3.client("s3")
        logger.info("Successfully created S3 client")

        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=bucket_path)
        logger.info(f"Successfully listed objects in bucket. Contents: {response.get('Contents', [])}")

        # Define regex patterns for file names
        date_pattern = re.compile(r"(\d{8})\.(csv|xlsx)$")
        forecast_pattern = re.compile(r"forecasts_(\d{4}-\d{2}-\d{2})\.csv$")
        evaluation_pattern = re.compile(r"evaluation_(\d{8})\.csv?$")
        confirmed_pattern = re.compile(r"confirmed_(\d{8})\.csv$")
        valid_files = []
        valid_main_files = []
        valid_forecast_files = []
        valid_evaluation_files = []
        valid_confirmed_files = []

        # Categorize files based on their names
        for obj in response.get("Contents", []):
            key = obj["Key"]
            match = date_pattern.search(key)
            forecast_match = forecast_pattern.search(key)
            evaluation_match = evaluation_pattern.search(key)
            confirmed_match = confirmed_pattern.search(key)
            if match:
                date_str = match.group(1)
                try:
                    file_date = datetime.strptime(date_str, "%d%m%Y")
                    if "main" in key:
                        valid_main_files.append((key, file_date))
                    elif "evaluation" not in key and "confirmed" not in key:
                        valid_files.append((key, file_date))
                    else:
                        pass
                except ValueError:
                    continue  #Skip files with invalid date format
            if forecast_match:
                forecast_date = datetime.strptime(forecast_match.group(1), "%Y-%m-%d")
                valid_forecast_files.append((key, forecast_date))
            if evaluation_match:
                eval_date_str = evaluation_match.group(1)
                try:
                    eval_file_date = datetime.strptime(eval_date_str, "%d%m%Y")
                    valid_evaluation_files.append((key, eval_file_date))
                except ValueError:
                    continue  # Skip files with invalid date format
            if confirmed_match:
                confirmed_date_str = confirmed_match.group(1)
                try:
                    confirmed_file_date = datetime.strptime(confirmed_date_str, "%d%m%Y")
                    valid_confirmed_files.append((key, confirmed_file_date))
                except ValueError:
                    continue  # Skip files with invalid date format

        if not valid_files and not valid_main_files and not valid_forecast_files and not valid_evaluation_files and not valid_confirmed_files:
            logger.warning("No valid files found in the S3 bucket")
            return None, None, None, None, None
        # Get the latest filenames for each category
        latest_file = (
            max(valid_files, key=lambda x: x[1])[0].split("/")[-1] if valid_files else None
        )
        latest_main_file = (
            max(valid_main_files, key=lambda x: x[1])[0].split("/")[-1]
            if valid_main_files
            else None
        )
        latest_forecast_file = (
            max(valid_forecast_files, key=lambda x: x[1])[0].split("/")[-1]
            if valid_forecast_files
            else None
        )
        latest_evaluation_file = (
            max(valid_evaluation_files, key=lambda x: x[1])[0].split("/")[-1]
            if valid_evaluation_files
            else None
        )
        latest_confirmed_file = (
            max(valid_confirmed_files, key=lambda x: x[1])[0].split("/")[-1]
            if valid_confirmed_files
            else None
        )

        return latest_file, latest_main_file, latest_forecast_file, latest_evaluation_file, latest_confirmed_file

    except Exception as e:
        logger.error(f"Error in get_latest_files: {str(e)}", exc_info=True)
        raise
# Add this function to test AWS credentials
def test_aws_credentials():
    try:
        sts = boto3.client('sts')
        response = sts.get_caller_identity()
        logger.info(f"AWS credentials are valid. Account ID: {response['Account']}")
    except Exception as e:
        logger.error(f"AWS credentials are invalid or not set: {str(e)}")

def run_pipeline(config_manager):
    """
    Run the main pipeline for data extraction, prediction, and evaluation.

    Args:
        config_manager (ConfigManager): The configuration manager object.
    """
    logger.info("Starting pipeline execution")
    
    # Add this at the beginning of the function
    test_aws_credentials()
    
    try:
        latest_file, latest_main_file, latest_forecast_file, latest_evaluation_file, latest_confirmed_file = get_latest_files(config_manager)
        logger.info(f"Latest file: {latest_file}")
        logger.info(f"Latest main file: {latest_main_file}")
        logger.info(f"Latest forecast file: {latest_forecast_file}")
        logger.info(f"Latest evaluation file: {latest_evaluation_file}")
        logger.info(f"Latest confirmed file: {latest_confirmed_file}")
        if latest_main_file is None:
            raise FileNotFoundError("Failed to retrieve the latest files from S3.")

        if not (latest_main_file.endswith(".csv")):
            raise ValueError(f"Invalid file format. Expected CSV files, got: {latest_file} and {latest_main_file}")
        
        s3_config = config_manager.s3_config
        extractor = Extractor(
            log_in_aws=True,
            bucket_path=s3_config['data_path'],
            bucket_name=s3_config['bucket_name'],
            original_file_name=latest_main_file,
            new_file_name=latest_file,
            client_name=s3_config['base_path']
        )

        
        full_data, newest_data = extractor.extract_data(client_name=config_manager.client_folder)
        if newest_data is not None:
            newest_data['date'] = pd.to_datetime(newest_data['date'])
        if full_data is not None:
            full_data['date'] = pd.to_datetime(full_data['date'])
        
        # Extract the client predictions
        #client_predictions = extractor.extract_predicted_data()
        #client_predictions = clean_clients_predictions_df(client_predictions)
        future_data = extractor.extract_predicted_data()
        
        #confirmed_df = extractor.extract_confirmed_data(latest_confirmed_file)
        #future_data, client_predictions = clean_clients_predictions_df(future_data)
        #future_data = clean_future_data(future_data, config_manager.get_config("DEMO_FORECAST_CONFIGURATIONS"))
        #client_predictions, confirmed_df = clean_clients_predictions_new(confirmed_df, future_data, full_data)
        #future_data = clean_future_data_new(future_data, config_manager.get_config("DEMO_FORECAST_CONFIGURATIONS"))
        

        # Extract the confirmed predictions 
        #if confirmed_df is not None:
            #confirmed_df = clean_clients_predictions_df(confirmed_df, return_predictions = False)
        
        if latest_forecast_file is not None:
            logger.info(f"Latest forecast file: {latest_forecast_file}")
            try:
                forecasted_data = extractor.extract_latest_predictions(latest_forecast_file)
            except Exception as e:
                logger.error(f"Error extracting latest predictions: {str(e)}", exc_info=True)
                forecasted_data = None


        # if we dont have the latest forecasts
        if (
            (latest_forecast_file is None)
            or (forecasted_data is None)
            or (full_data is not None and np.any(np.isin(full_data["date"].unique(), forecasted_data["date"].unique())))
        ):
            logger.info("Running new prediction")
            try:
                predictions = run_prediction(full_data,  config_manager)
                logger.info("New forecasts generated successfully")
            except Exception as e:
                logger.error(f"Error generating forecasts: {str(e)}", exc_info=True)
                raise

            predictions = concat_with_true_values(predictions, newest_data)

            predictions_json = predict_jsonify(predictions)
            save_predictions(predictions, predictions_json, config_manager)
        
        # if we have forecasts but new values have been uplaoded for evalaution
        elif ((forecasted_data is not None) 
        and (np.any(np.isin(forecasted_data["date"].unique(), newest_data["date"].unique())))
        and (forecasted_data['True'].isna().all())
        ):
            
            logger.info("Loading predictions")
            predictions = forecasted_data.copy()

            predictions = concat_with_true_values(predictions, newest_data)
            predictions_json = predict_jsonify(predictions)
            save_predictions(predictions, predictions_json, config_manager)
            
            if not predictions['True'].isna().all():
                logger.info("Running evaluation")
                plot_filenames = run_evaluation(full_data, predictions, config_manager, latest_evaluation_file)
            else:
                logger.info("Skipping plots as no previous forecast data is available.")
                plot_filenames = []
        
        # If the pipeline is not needed
        else:
            logger.info("Pipeline not needed")
            predictions = None
            plot_filenames = []

    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}", exc_info=True)
        raise
    except ValueError as e:
        logger.error(f"Value error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_pipeline: {str(e)}", exc_info=True)
        raise

    logger.info("Pipeline execution completed")


def clean_future_data(future_data, forecast_config):
    """
    Clean the future data.

    Args:
        future_data (pd.DataFrame): The future data.

    Returns:
        pd.DataFrame: The cleaned future data.
    """
    #future_data = add_extra_covariates(future_data)
    
    preprocessor = DataPreprocessor(future_data)

    future_data = preprocessor.preprocess(remove_weekends=forecast_config["remove_weekends"],
                                            #item_covars=forecast_config["item_covars"],
                                            #   item_time_covars=forecast_config["item_time_covars"]
                                            )

    future_data = future_data.rename(columns = {'date': 'ds'})
    #future_data['Holiday'] = future_data['Holiday'].astype("category").cat.codes.astype("category")
    future_data = future_data[['unique_id', 'ds'] + forecast_config["item_time_covars"]]

    return future_data


def add_extra_covariates(df):
    """
    Add the extra covariates to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the covariates to.

    Returns:
        pd.DataFrame: The dataframe with the added covariates.
    """

    df['School'] = df['unique_id'].str.split('_').str[0]
    # On customer 2 if the last letter is not int remove it
    df['School'] = df['School'].apply(lambda x: x if x.isdigit() else x[:-1])

    df['Size'] = df['unique_id'].str.split('_').str[1]

    df['Type'] = df['unique_id'].str.split('_').str[2]

    df['Meal'] = df['ProteinGroup'] + '_' + df['VegetablesGroup'] + '_' + df['StarchGroup']
        
    return df


def run_prediction(full_data, config_manager):
    """
    Run the prediction pipeline.

    Args:
        full_data (pd.DataFrame): The full dataset to use for prediction.
        future_data (pd.DataFrame): The future data to use for prediction.
        client_predictions (pd.DataFrame): The client predictions to use for prediction.
        config_manager (ConfigManager): The configuration manager object.
    """
    logger.info("Starting prediction process")
    logger.info(f"Memory usage before prediction: {psutil.virtual_memory().percent}%")

    # Process the data

    # Predict -> assuming hyperparams are tuned
    model_config = config_manager.get_config("DEMO_MODEL_CONFIGURATIONS")
    forecast_config = config_manager.get_config("DEMO_FORECAST_CONFIGURATIONS")
    benchmark_config = config_manager.get_config("DEMO_BENCHMARK_CONFIGURATIONS")
    
    # Add the extra covariates here.
    #full_data = add_extra_covariates(full_data)
    preprocessor = DataPreprocessor(full_data)
    full_data = preprocessor.preprocess(remove_weekends=forecast_config["remove_weekends"],
                                          #item_covars=forecast_config["item_covars"],
                                          #item_time_covars=forecast_config["item_time_covars"],
                                          filter_dates=True)

    forecaster = Forecaster(
        model_config=model_config,
        forecast_config=forecast_config,
        seasonal_length=forecast_config["seasonal_length"]
    )
    forecaster.fit(full_data)

    # Get the predictions
    predictions = forecaster.predict()
    
    # Post processes
    predictions = postprocess_predictions(predictions,  ensemble = True)
    
    logger.info(f"Memory usage after prediction: {psutil.virtual_memory().percent}%")
    logger.info("Prediction process completed")
    return predictions


def run_evaluation(full_data, predictions, config_manager, latest_evaluation_file):
    """
    Run the evaluation pipeline.

    Args:
        full_data (pd.DataFrame): The full dataset.
        predictions (pd.DataFrame): The predictions dataframe.
        config_manager (ConfigManager): The configuration manager object.
        latest_evaluation_file (str): The latest evaluation file name.
    """
    logger.info("Starting evaluation process")
    # Define the evaluator
    predictions['cv'] = 1

    ids = predictions[(predictions['Model'] == 'SeasonXpert') & (predictions['y'].isna())]['unique_id'].unique()

    # Filter out rows with nans in y or True
    predictions = predictions[~predictions['unique_id'].isin(ids)]
    predictions = predictions[~predictions['y'].isna()]
    predictions = predictions[~predictions['True'].isna()]

    eval = ForecastEvaluator(freq="D", original_df=full_data)
    # Fit
    eval.fit(complete_evaluation_df =predictions,  evaluation_cv=True)
    # Predict
    evaluation_config = config_manager.get_config("DEMO_EVALUATION_CONFIGURATIONS")
    eval_df = eval.predict(metrics=evaluation_config["metrics"])
    # Update the evaluations
    full_evaluation_df = update_evaluations(config_manager, eval_df, latest_evaluation_file)
    if full_evaluation_df is not None:
        # Generate and upload plots
        plot_filenames = make_plots(full_evaluation_df, config_manager)
        logger.info(f"Generated plot filenames:")
    else:
        logger.warning("No evaluation data available for plotting")
        plot_filenames = []

    logger.info("Evaluation completed")
    return plot_filenames

def update_evaluations(config_manager, new_eval_df, latest_evaluation_file):
    """
    Load and concatenate the evaluations from S3, then save the updated file.

    Args:
        config_manager (ConfigManager): The configuration manager object.
        new_eval_df (pd.DataFrame): The new evaluation dataframe.
        latest_evaluation_file (str): The latest evaluation file name.

    Returns:
        pd.DataFrame: The concatenated evaluation dataframe.
    """
    logger.info("Starting to update evaluations")

    # Get S3 configuration
    s3_config = config_manager.get_s3_config()
    logger.info(f"S3 config: {s3_config}")

    # Create S3 client
    s3 = boto3.client('s3')

    try:
        # Step 1: Load the latest evaluation file from S3
        if latest_evaluation_file:
            logger.info(f"Loading latest evaluation file: {latest_evaluation_file}")
            response = s3.get_object(Bucket=s3_config['bucket_name'], Key=f"{s3_config['data_path']}/{latest_evaluation_file}")
            latest_eval_df = pd.read_csv(io.BytesIO(response['Body'].read()))
        else:
            logger.info("No previous evaluation file found. Creating new one.")
            latest_eval_df = pd.DataFrame()

        # Step 2: Concatenate the latest evaluation data with the new evaluation data
        combined_eval_df = pd.concat([latest_eval_df, new_eval_df], ignore_index=True)

        # Step 3: Get the max date from the new evaluation data for the new filename
        new_eval_df['date'] = pd.to_datetime(new_eval_df['date'])
        max_date = new_eval_df['date'].max().strftime('%d%m%y')

        # Step 4: Save the combined evaluation data to S3
        csv_buffer = io.StringIO()
        combined_eval_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        new_eval_key = f"{s3_config['data_path']}/evaluation_{max_date}.csv"
        logger.info(f"Saving updated evaluation file to S3: {new_eval_key}")
        s3.put_object(
            Bucket=s3_config['bucket_name'],
            Key=new_eval_key,
            Body=csv_content
        )
        logger.info("Updated evaluation file saved successfully")

        return combined_eval_df

    except ClientError as e:
        logger.error(f"An error occurred while updating evaluations in S3: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while updating evaluations: {e}")

    logger.info("Finished updating evaluations")
    return None  # Return None if an error occurred

def save_predictions(predictions, predictions_json, config_manager):
    """
    Save the predictions to S3.

    Args:
        predictions (pd.DataFrame): The predictions dataframe.
        predictions_json (dict): The jsonified predictions.
        config_manager (ConfigManager): The configuration manager object.
    """
    logger.info("Starting to save predictions")
    
    try:
        # Take the max date from the predictions
        max_date = predictions["date"].max()

        # Get S3 configuration
        s3_config = config_manager.get_s3_config()
        logger.info(f"S3 config: {s3_config}")

        # Create S3 client
        s3 = boto3.client('s3')

        # Save the jsonified predictions to S3
        json_key = f"{s3_config['model_path']}/latest_predictions.json"
        logger.info(f"Attempting to save JSON to S3: {json_key}")
        s3.put_object(
            Bucket=s3_config['bucket_name'],
            Key=json_key,
            Body=json.dumps(predictions_json)
        )
        logger.info(f"Jsonified predictions saved to S3: {json_key}")

        # Save the predictions df to S3
        csv_buffer = io.StringIO()
        predictions.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        csv_key = f"{s3_config['data_path']}/forecasts_{max_date}.csv"
        logger.info(f"Attempting to save CSV to S3: {csv_key}")
        s3.put_object(
            Bucket=s3_config['bucket_name'],
            Key=csv_key,
            Body=csv_content
        )
        logger.info(f"Predictions CSV saved to S3: {csv_key}")
        
    except ClientError as e:
        logger.error(f"An error occurred while saving to S3: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    logger.info("Finished saving predictions")


def set_extra_days_zeros(df, extra_days_path):
    """
    Set the values of the Wednesday forecasts to 0.

    Args:
        df (pd.DataFrame): The dataframe to modify.
        extra_days_path (str): The path to the dataframe containing the extra days.

    Returns:
        pd.DataFrame: The modified dataframe.
    """
    
    # merge on unique_id
    extra_days = pd.read_csv(extra_days_path)
    temp = df.copy()
    temp = temp.merge(extra_days, on = 'unique_id', how = 'left')

    # Extract the day from the date
    temp['day'] = temp['date'].dt.day_name()
    # if it is Wednesday and ExtraDay is 1, set the value of y to 0
    temp.loc[(temp['day'] == 'Wednesday') & (temp['ExtraDay'] == 0), 'y'] = 0

    # clean columns
    temp = temp.drop(columns = ['day', 'ExtraDay'])
    return temp



def get_ensemble_predictions(predictions, ensemble=True, double_step = True):
    cols = ['unique_id', 'date', 'fh']
    def custom_agg(x, min_ratio = 2, medium_ratio = 5, max_ratio = 7):
            non_zero = x[x != 0]
            if len(non_zero) == 0:
                return 0
            elif len(non_zero) == 1:
                return non_zero.iloc[0]
            else:
                min_val, max_val = min(non_zero), max(non_zero)
                ratio = max_val / min_val
                
                # Dynamic ratio threshold based on the magnitude of predictions
                if min_val < 10:
                    ratio_threshold = min_ratio
                elif min_val < 50:
                    ratio_threshold = medium_ratio
                else:
                    ratio_threshold = max_ratio
                
                if ratio > ratio_threshold:
                    return max_val
                else:
                    return non_zero.median()
    if double_step:
        if len(predictions['Model'].unique()) > 2:
            # First we merge the two benchmarks
            benchmark_predictions = predictions[predictions['Model']!='lgbm']
            # set min_ratio = 2, medium_ratio = 4, max_ratio = 6
            benchmark_predictions = benchmark_predictions.groupby(cols).agg({'y': lambda x: custom_agg(x, min_ratio=1, medium_ratio=1, max_ratio=1)}).reset_index()
            benchmark_predictions['Model'] = 'SeasonalEnsemble'
            predictions = pd.concat([predictions, benchmark_predictions], ignore_index=True)
            
    if ensemble:
        

        ensemble_preds = predictions.groupby(cols).agg({'y': custom_agg}).reset_index()
    else:
        ensemble_preds = predictions.copy()

    ensemble_preds['Model'] = 'SeasonXpert'

    return ensemble_preds


def clean_clients_predictions_df(predictions, return_predictions = True):
    """
    Clean the client predictions dataframe.

    Args:
        predictions (pd.DataFrame): The predictions dataframe.

    Returns:
        pd.DataFrame: The cleaned predictions dataframe.
    """
    if "BEVESTIGD" in predictions.columns:
        predictions = predictions.drop(columns = ['BEVESTIGD'])
    predictions = predictions.drop('Postcode', axis = 1)
    
    cleaned_preds_full = preprocess_pipeline(predictions, clear_zip_code=False, clean_extra_day=True)
    cleaned_preds = cleaned_preds_full[['date', 'Customer', 'y']]
    cleaned_preds = cleaned_preds.rename(columns = {'Customer': 'unique_id'})
    if return_predictions:
        cleaned_preds['Model'] = 'Vision'
        # cleaned_preds_full = cleaned_preds_full.rename(columns = {'date': 'ds'})
        return cleaned_preds_full, cleaned_preds
    else:
        cleaned_preds['Model'] = 'Confirmed'
        return cleaned_preds


def merge_predictions_with_benchmark(predictions_df, benchmark_df):
    """
    Add missing unique_ids from benchmark_df to predictions_df with NaN predictions,
    and add forecast horizons (fh) to the benchmark dataframe.
    
    Args:
    predictions_df (pd.DataFrame): DataFrame with original predictions
    benchmark_df (pd.DataFrame): DataFrame with benchmark predictions
    
    Returns:
    pd.DataFrame: Updated predictions DataFrame with added unique_ids and forecast horizons
    """
    # Get the unique_ids that are in benchmark_df but not in predictions_df
    # just concat

    return pd.concat([predictions_df, benchmark_df], ignore_index=True)



def estimate_mean_difference(predictions):
    pivoted = predictions.pivot_table(
    values='y', 
    index=['unique_id', 'date', 'fh'], 
    columns='Model', 
    aggfunc='first'
        ).reset_index()

    # Calculate the absolute difference
    pivoted['AbsDifference'] = np.abs(pivoted['SeasonXpert'] - pivoted['SeasonalNaive'])

    # Sort the dataframe
    result = pivoted.sort_values(['unique_id', 'fh', 'date']).reset_index(drop=True)
    # drop nans
    result = result.dropna()
    result = result.groupby('unique_id').mean().reset_index()[['unique_id', 'AbsDifference']]

    result['Color'] = result['AbsDifference'].apply(get_color)
    return result[['unique_id', 'Color']]

def get_color(value):
    if value < 20:
        return 'green'
    elif 20 <= value < 40:
        return 'yellow'
    else:
        return 'red'
    

def concat_with_true_values(predictions, newest_data):
    """
    Concatenate the predictions with the true values.

    Args:
        predictions (pd.DataFrame): The predictions dataframe.
        newest_data (pd.DataFrame): The newest data dataframe.

    Returns:
        pd.DataFrame: The concatenated dataframe.
    newest_data
    """
    if newest_data is None:
        cleaned_merged = predictions.copy()
        cleaned_merged['True'] = np.nan
    else:
        
        if (predictions['date'].unique() == newest_data['date'].unique()).all():
            if 'True' in predictions.columns:
                predictions = predictions.drop(columns = ['True'])
            clean_new_data = newest_data[['date', 'unique_id', 'y']]
            clean_new_data = clean_new_data.rename(columns = {'Customer': 'unique_id',
                                                            'y':'True'})
            # merge 
            cleaned_merged = predictions.merge(clean_new_data, on = ['date', 'unique_id'], how = 'left')
        else:
            print("Dates are different")
            cleaned_merged = predictions.copy()
            cleaned_merged['True'] = np.nan 
    
    return cleaned_merged


def postprocess_predictions(predictions, ensemble = False, double_step = True):
    # set the extra days to zero
    #predictions = set_extra_days_zeros(predictions, wednesdays_off_path)

    # clip predictions to greater or equal to 0
    predictions['y'] = predictions['y'].clip(lower=0)
    predicted_data = predictions[predictions['Model'] == 'SeasonalNaive']

    # get the ensemble predictions
    predictions = get_ensemble_predictions(predictions, ensemble = ensemble, double_step=double_step)

    # round the predictions 
    predictions['y'] = predictions['y'].round(0).astype(int)

    # Keep only data on the predictions
    #predictions = predictions[predictions['unique_id'].isin(predicted_data['unique_id'].unique())]
    
    # add missing unique ids
    predictions = merge_predictions_with_benchmark(predictions, predicted_data)

    differences = estimate_mean_difference(predictions)

    predictions = pd.merge(predictions, differences, on = 'unique_id', how = 'left')

    return predictions


def merge_predictions_with_confirmed(predictions_df, confirmed_df):
    # Ensure date column in predictions_df is in datetime format
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    # Create placeholder confirmed rows
    placeholder_df = predictions_df[['unique_id', 'date', 'fh']].drop_duplicates()
    placeholder_df['Model'] = 'Confirmed'
    placeholder_df['Color'] = 'green'
    placeholder_df['y'] = np.nan

    # Check if confirmed_df is None or dates don't match
    if confirmed_df is None or not set(pd.to_datetime(confirmed_df['date'])).issubset(set(predictions_df['date'])):
        # Use placeholder data
        merged_df = pd.concat([predictions_df, placeholder_df], ignore_index=True)
    else:
        # Dates match, proceed with original approach
        confirmed_df['date'] = pd.to_datetime(confirmed_df['date'])
        confirmed_df['Model'] = 'Confirmed'
        confirmed_df['Color'] = 'green'
        
        # Create a date to fh mapping from the original predictions
        date_fh_mapping = predictions_df.groupby('date')['fh'].first().to_dict()
        confirmed_df['fh'] = confirmed_df['date'].map(date_fh_mapping)
        
        merged_df = pd.concat([predictions_df, confirmed_df], ignore_index=True)
    
    # Sort the dataframe
    merged_df = merged_df.sort_values(['unique_id', 'date', 'Model']).reset_index(drop=True)
    
    return merged_df

def predict_jsonify(predictions):
    """
    Predict the future values and format them as JSON, including True values.
    """
    logger.info("Starting JSON prediction process")
    

    # Include 'True' in the columns to group
    cols_to_group = ["date", "y", "Model", "Color", "True"]
    
    predictions["date"] = predictions["date"].dt.date.astype(str)

    jsonified_predictions = (
        predictions.groupby("unique_id")
        .apply(lambda x: x[cols_to_group].to_dict("records"))
        .to_dict()
    )

    logger.info("JSON prediction process complete")
    return jsonified_predictions


def create_identifier(row):
    return f"{row['Datum']}_{row['Rit']}_{row['Hoeveelheidstype']}_{row['Verpakkingstype']}_{row['Menugroep']}_{row['Menusoort']}"

def merge_new_future_predictions(predicted_latest_data, future_data_complete):
    # Convert 'Datum' to datetime if it's not already
    predicted_latest_data['Datum'] = pd.to_datetime(predicted_latest_data['Datum'])
    future_data_complete['Datum'] = pd.to_datetime(future_data_complete['Datum'])

    # Get the date range of predicted_latest_data
    start_date = predicted_latest_data['Datum'].min()
    end_date = predicted_latest_data['Datum'].max()

    # Filter future_data_complete for the same date range
    future_data_filtered = future_data_complete[(future_data_complete['Datum'] >= start_date) & (future_data_complete['Datum'] <= end_date)]

    # Create a unique identifier for each row
    predicted_latest_data['identifier'] = predicted_latest_data.apply(create_identifier, axis=1)
    future_data_filtered['identifier'] = future_data_filtered.apply(create_identifier, axis=1)

    # Find rows in future_data_filtered that are not in predicted_latest_data
    missing_rows = future_data_filtered[~future_data_filtered['identifier'].isin(predicted_latest_data['identifier'])]

    # Combine predicted_latest_data with missing rows
    updated_predicted_latest_data = pd.concat([predicted_latest_data, missing_rows], ignore_index=True)

    # Remove the temporary identifier column
    updated_predicted_latest_data = updated_predicted_latest_data.drop('identifier', axis=1)


    return updated_predicted_latest_data


def clean_clients_predictions_new(confirmed_weekly_data, future_data_complete, full_data):
    """
    Clean the client predictions dataframe.

    Args:
        confirmed_weekly_data (pd.DataFrame): The confirmed weekly data.
        future_data_complete (pd.DataFrame): The future data complete.

    Returns:
        pd.DataFrame: The cleaned predictions dataframe.
    """
    full_data['date'] = pd.to_datetime(full_data['date'])
    future_final_date = full_data['date'].max() + pd.DateOffset(weeks=1)
    confirmed_weekly_data['Datum'] = pd.to_datetime(confirmed_weekly_data['Datum'])
    max_confirmed_data_date = confirmed_weekly_data['Datum'].max()
    
    # if we have confirmed data
    if (confirmed_weekly_data is not None) and (max_confirmed_data_date == future_final_date):
        print('confirm')
        # Split predictiosn and confirmed 
        confirmed_methods = ['Website', 'Telefoon']
        to_drop = ['BEVESTIGD', 'Postcode']
        client_confirmations = confirmed_weekly_data[confirmed_weekly_data['BEVESTIGD'].isin(confirmed_methods)].drop(columns = to_drop)
        predicted_latest_data = confirmed_weekly_data[~confirmed_weekly_data['BEVESTIGD'].isin(confirmed_methods)].drop(columns = to_drop)

        # Update predicted data with the missing rows
        predicted_latest_data = merge_new_future_predictions(predicted_latest_data, future_data_complete)
        predicted_latest_data = predicted_latest_data.drop(columns = ['Postcode'])
    
        predicted_latest_data = preprocess_pipeline(predicted_latest_data, clear_zip_code=False, clean_extra_day=True)
        predicted_latest_data = predicted_latest_data[['date', 'Customer', 'y']]
        predicted_latest_data = predicted_latest_data.rename(columns = {'Customer': 'unique_id'})
        predicted_latest_data['Model'] = 'Vision'
    
        client_confirmations = preprocess_pipeline(client_confirmations, clear_zip_code=False, clean_extra_day=True)
        client_confirmations = client_confirmations[['date', 'Customer', 'y']]
        client_confirmations = client_confirmations.rename(columns = {'Customer': 'unique_id'})
        client_confirmations['Model'] = 'Confirmed'

    # otherwise we take full predictions
    else:
        predicted_latest_data = future_data_complete.drop(columns = ['Postcode'])
        predicted_latest_data = preprocess_pipeline(predicted_latest_data, clear_zip_code=False, clean_extra_day=True)
        predicted_latest_data = predicted_latest_data[['date', 'Customer', 'y']]
        predicted_latest_data = predicted_latest_data.rename(columns = {'Customer': 'unique_id'})
        predicted_latest_data['Model'] = 'Vision'
        predicted_latest_data = predicted_latest_data.sort_values(['unique_id', 'date']).reset_index(drop=True)
        client_confirmations = None

    return predicted_latest_data, client_confirmations

def clean_future_data_new(future_data, forecast_config):
    """
    Clean the future data.

    Args:
        future_data (pd.DataFrame): The future data.

    Returns:
        pd.DataFrame: The cleaned future data.
    """
    
    if "BEVESTIGD" in future_data.columns:
        future_data = future_data.drop(columns = ['BEVESTIGD'])
    future_data = future_data.drop('Postcode', axis = 1)
    
    cleaned_future_data = preprocess_pipeline(future_data, clear_zip_code=False, clean_extra_day=True)

    cleaned_future_data = add_extra_covariates(cleaned_future_data)
    preprocessor = DataPreprocessor(cleaned_future_data)

    cleaned_future_data = preprocessor.preprocess(remove_weekends=forecast_config["remove_weekends"],
                                            item_covars=forecast_config["item_covars"],
                                            item_time_covars=forecast_config["item_time_covars"])

    cleaned_future_data = cleaned_future_data.rename(columns = {'date': 'ds'})
    #future_data['Holiday'] = future_data['Holiday'].astype("category").cat.codes.astype("category")
    cleaned_future_data = cleaned_future_data[['unique_id', 'ds'] + forecast_config["item_time_covars"]]

    return cleaned_future_data