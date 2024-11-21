from logs.logging_config import setup_logger
import pandas as pd
import os
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import BotoCoreError, ClientError
#from modules.utils.aws_config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from io import StringIO
from datetime import datetime
from modules.data_preprocessor import DataPreprocessor
import io
from dotenv import load_dotenv

load_dotenv()

# Set up logger
logger = setup_logger(__name__, 'data_extractor.log')
class Extractor:
    """
    A class to extract the clients data from AWS S3.
    """

    def __init__(
        self,
        temp_path: str = None,
        bucket_name: str = None,
        original_file_name: str = None,
        new_file_name: str = None,
        log_in_aws: bool = False,
        bucket_path: str = None,
        client_name: str = None,
    ) -> None:
        """
        Initialize the Extractor with the S3 bucket name and file name.

        Args:
            temp_path (str): The path to the temporary file for testing
            bucket_name (str): The name of the S3 bucket.
            original_file_name (str): The name of the original file with the training data
            new_file_name (str): The name of the new file arriving each week
            forecast_file_name (str): The name of the forecast file
            log_in_aws (boolean): Whether to log in to AWS
            bucket_path (str): The path to the bucket
        """

        # assert either the path or the bucket name have been given
        assert (
            temp_path or bucket_name
        ), "Either the path or the bucket name must be given"
        

        self.original_file_name = original_file_name
        self.new_file_name = new_file_name
        self.path = temp_path
        self.client_name = client_name
        self.log_in_aws = log_in_aws

        if log_in_aws:
            self.bucket_name = bucket_name
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            self.bucket_path = bucket_path

        logger.info(f"Extractor initialized with bucket: {bucket_name}, path: {bucket_path}")



    def extract_future_data(self, file_name: str = "future_meal_holidays.csv") -> pd.DataFrame:
        """
        Extract the future data from the S3 bucket.

        Args:
            file_name (str): The name of the file with the future data.

        Returns:
            pd.DataFrame: The extracted data.
        """
        
        logger.info(f"Extracting future data from {file_name}")
        try:
            if self.log_in_aws:
                full_path = self.bucket_path + "/" + file_name
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_path)
                data = pd.read_csv(obj["Body"])
            else:
                full_path = self.path + "/" + file_name
                data = pd.read_csv(full_path)

            logger.info(f"Future data extracted successfully from {full_path}")
            # rename to match with mlforecast 
            data = data.rename(columns = {'date': 'ds'})
            return data
        except Exception as e:
            logger.error(f"Failed to extract future data: {e}")
            return pd.DataFrame()
        


    def extract_predicted_data(self, file_name: str = "client_1_october_original.csv") -> pd.DataFrame:
        """
        Extract the predicted data from the S3 bucket.

        Args:
            file_name (str): The name of the file with the predictions.

        Returns:
            pd.DataFrame: The extracted data.
        """

        logger.info(f"Extracting predicted data from {file_name}")
        try:
            if self.log_in_aws:
                full_path = self.bucket_path + "/" + file_name
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_path)
                data = pd.read_csv(obj["Body"])
            else:
                full_path = self.path + "/" + file_name
                data = pd.read_csv(full_path)
            logger.info(f"Predicted data extracted successfully from {full_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to extract predicted data: {e}")
            return pd.DataFrame()
                
    def extract_original_data(self) -> pd.DataFrame:
        """
        Extract the original training data from the S3 bucket.

        Returns:
            pd.DataFrame: The extracted data.
        """
        logger.info("Extracting original data")
        try:
            if self.log_in_aws:
                full_path = self.bucket_path + "/" + self.original_file_name
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_path)
                data = pd.read_csv(obj["Body"])
            else:
                full_path = self.path + "/" + self.original_file_name
                data = pd.read_csv(full_path)
            logger.info(f"Original data extracted successfully from {full_path}")
            return data
        except (BotoCoreError, ClientError) as error:
            logger.error(f"Failed to fetch original data: {error}")
            return pd.DataFrame()

    def extract_new_data(self, client_name: str = None) -> pd.DataFrame:
        """
        Extract the new data from the S3 bucket.

        Args:
            client_name (str): The name of the client

        Returns:
            pd.DataFrame: The extracted data.
        """
        logger.info("Extracting new data")
        if self.log_in_aws:
            full_path = self.bucket_path + "/" + self.new_file_name
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_path)
            content = obj["Body"].read()

            # Determine file type based on extension
            file_extension = self.new_file_name.split('.')[-1].lower()

            try:
                if file_extension == 'csv':
                    # First, try reading with default settings
                    data = pd.read_csv(io.BytesIO(content))
                    
                    # If we get a single column, try reading with semicolon separator
                    if data.shape[1] == 1:
                        data = pd.read_csv(io.BytesIO(content), sep=';')
                elif file_extension in ['xls', 'xlsx']:
                    data = pd.read_excel(io.BytesIO(content))
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                if 'm5' not in self.client_name:
                    dp = DataPreprocessor(data)
                    data = dp.preprocess_new_data(client_name="client_1", new_file=True)
                    logger.info(f"New data extracted successfully from {full_path}")
                return data
            except Exception as e:
                logger.error(f"Failed to extract new data: {e}")
                return pd.DataFrame()
        else:
            full_path = self.path + "/" + self.new_file_name
            file_extension = self.new_file_name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                data = pd.read_csv(full_path)
            elif file_extension in ['xls', 'xlsx']:
                data = pd.read_excel(full_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            return data
        

    def extract_data(self, client_name: str = None) -> pd.DataFrame:
        """
        Extract the data from the S3 bucket or local storage.

        Args:
            client_name (str): The name of the client

        Returns:
            pd.DataFrame: The extracted data.
            pd.DataFrame: The new data (if any).
        """
        logger.info("Extracting data")
        self.original = self.extract_original_data()

        if self.new_file_name is not None:
            self.new_date = datetime.strptime(self.new_file_name[:6], "%d%m%y")
            self.new = self.extract_new_data(self.client_name)
        else:
            self.new_date = None
            self.new = None
        
        #if self.log_in_aws:
            #original_date = datetime.strptime(self.original_file_name[5:11], "%d%m%y")
            
            
            #if (original_date == new_date) or (new_date is None):
            #    logger.info("No new data to update")
            #   return original, new
            #else:
            #    logger.info("Updating data with new entries")
            #    newly_arrived_data = pd.concat([original, new], ignore_index=True)
            #    newly_arrived_data['date'] = pd.to_datetime(newly_arrived_data['date'])
            #    self.update_save_data(newly_arrived_data)
            #    return newly_arrived_data, new
        #else:
        return self.original, self.new


    def merge_new_data(self) -> pd.DataFrame:
        """
        Merge the new data with the original data.

        Args:
            original (pd.DataFrame): The original data.
            new (pd.DataFrame): The new data.

        Returns:
            pd.DataFrame: The merged data.
        """
        original_date = datetime.strptime(self.original_file_name[5:11], "%d%m%y")
            
        if (original_date == self.new_date) or (self.new_date is None):
            logger.info("No new data to update")
        else:
            logger.info("Updating data with new entries")
            self.original['date'] = pd.to_datetime(self.original['date'])
            newly_arrived_data = pd.concat([self.original, self.new], ignore_index=True)
            newly_arrived_data['date'] = pd.to_datetime(newly_arrived_data['date'])
            self.update_save_data(newly_arrived_data)


    def update_save_data(self, df) -> None:
        """
        Save the new data to the S3 bucket or locally with the format main_{latest_date_without_main}.

        Args:
            df (pd.DataFrame): The new data to save.
        """
        latest_date = self.new_file_name[:8]
        new_file_name = f"main_{latest_date}.csv"
        logger.info(f"Saving updated data as {new_file_name}")

        try:
            if self.log_in_aws:
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                full_path = f"{self.bucket_path}/{new_file_name}"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=full_path,
                    Body=csv_buffer.getvalue()
                )
            else:
                full_path = os.path.join(self.path, new_file_name)
                df.to_csv(full_path, index=False)
            logger.info(f"Data saved successfully to {full_path}")
        except Exception as e:
            logger.error(f"Failed to save updated data: {e}")

    def extract_latest_predictions(self, file_name: str) -> pd.DataFrame:
        """
        Extract the latest predictions from the S3 bucket.

        Args:
            file_name (str): The name of the file with the predictions.

        Returns:
            pd.DataFrame: The extracted data.
        """
        logger.info(f"Extracting latest predictions from {file_name}")
        try:
            if self.log_in_aws:
                full_path = self.bucket_path + "/" + file_name
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_path)
                data = pd.read_csv(obj["Body"])
                data['date'] = pd.to_datetime(data['date'])
            else:
                full_path = self.path + "/" + file_name
                data = pd.read_csv(full_path)
                data['date'] = pd.to_datetime(data['date'])
            logger.info("Latest predictions extracted successfully")
            return data
        except Exception as e:
            logger.error(f"Failed to extract latest predictions: {e}")
            return pd.DataFrame()

    def extract_confirmed_data(self, file_name: str = None) -> pd.DataFrame:
        """
        Extract the confirmed data from the S3 bucket.

        Args:
            file_name (str): The name of the file with the confirmed data.

        Returns:
            pd.DataFrame: The extracted data.
        """
        logger.info(f"Extracting confirmed data from {file_name}")
        try:
            if self.log_in_aws:
                full_path = self.bucket_path + "/" + file_name
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_path)
                data = pd.read_csv(obj["Body"])
            elif file_name is None:
                logger.info("No confirmed data to extract")
                return None
            else:
                full_path = self.path + "/" + file_name
                data = pd.read_csv(full_path)
            logger.info(f"Confirmed data extracted successfully from {full_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to extract confirmed data: {e}")
            return pd.DataFrame()

def handle_file_upload(file, bucket_name, bucket_path):
    """
    Handle the file upload process to AWS S3 independently.

    Args:
        file: File object to upload.
        bucket_name (str): The name of the S3 bucket.
        bucket_path (str): The path within the S3 bucket.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    logger.info(f"Handling file upload: {file.filename}")
    if file and file.filename:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            full_path = f"{bucket_path}/{filename}"
            s3_client.upload_file(file_path, bucket_name, full_path)
            
            os.remove(file_path)
            logger.info(f"File {filename} uploaded successfully to S3")
            return True
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS error uploading file: {e}")
            os.remove(file_path)
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            os.remove(file_path)
            return False
    logger.warning("No file provided for upload")
    return False
