import importlib
import boto3
from typing import Dict, Any, Tuple
from modules.utils.user_configurations import PASSWORDS
import os

class ConfigManager:
    """
    A class to manage client configurations and S3 interactions.

    This class loads client-specific configurations and sets up S3 access for data storage and retrieval.
    """

    def __init__(self, bucket_name: str, client_folder: str):
        """
        Initialize the ConfigManager with S3 bucket and client folder information.

        Args:
            bucket_name (str): Name of the S3 bucket
            client_folder (str): Name of the client-specific folder
        """
        self.bucket_name = bucket_name
        self.client_folder = client_folder
        self.s3 = boto3.client('s3',
                               aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                               aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                               region_name=os.environ.get('AWS_DEFAULT_REGION'))
        self.client_config, self.s3_config = self._load_configs()

    def _load_configs(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Load client-specific configurations and set up S3 paths.

        Returns:
            Tuple[Dict[str, Any], Dict[str, str]]: Client config and S3 config dictionaries
        """
        try:
            # Dynamically import client-specific configuration module
            config_module = importlib.import_module(f'modules.utils.configurations.{self.client_folder}')
            
            # Extract relevant configurations from the imported module
            client_config = {
                "DEMO_FORECAST_CONFIGURATIONS": getattr(config_module, "DEMO_FORECAST_CONFIGURATIONS", None),
                "DEMO_MODEL_CONFIGURATIONS": getattr(config_module, "DEMO_MODEL_CONFIGURATIONS", None),
                "DEMO_BENCHMARK_CONFIGURATIONS": getattr(config_module, "DEMO_BENCHMARK_CONFIGURATIONS", None),
                "DEMO_EVALUATION_CONFIGURATIONS": getattr(config_module, "DEMO_EVALUATION_CONFIGURATIONS", None),
                "DEMO_AWS_LOAD_CONFIGURATIONS": getattr(config_module, "DEMO_AWS_LOAD_CONFIGURATIONS", None),
                "FEATURE_ENGINEERING_CONFIG": getattr(config_module, "FEATURE_ENGINEERING_CONFIG", None),
                "HYPERPARAMETER_SEARCH_CONFIG": getattr(config_module, "HYPERPARAMETER_SEARCH_CONFIG", None),
                "HYPERPARAMETER_TOTAL_SEARCH_CONFIG": getattr(config_module, "HYPERPARAMETER_TOTAL_SEARCH_CONFIG", None)
            }
            
            # Set up S3 paths for various data storage needs
            s3_config = {
                "bucket_name": self.bucket_name,
                "base_path": self.client_folder,
                "data_path": f"{self.client_folder}/data",
                "plots_path": f"{self.client_folder}/plots",
                "configurations_path": f"{self.client_folder}/configurations",
                "model_path": f"{self.client_folder}/model"
            }
            
            return client_config, s3_config
        except ImportError:
            # Raise an error if the client-specific configuration file is not found
            raise ValueError(f"Configuration file not found for client: {self.client_folder}")
        except AttributeError as e:
            # Raise an error if there's an issue with the configuration file structure
            raise ValueError(f"Error loading configurations for client {self.client_folder}: {str(e)}")

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Retrieve a specific configuration by name.

        Args:
            config_name (str): Name of the configuration to retrieve

        Returns:
            Dict[str, Any]: The requested configuration or None if not found
        """
        config = self.client_config.get(config_name)
        if config is None:
            print(f"Warning: Configuration '{config_name}' not found.")
        return config

    def get_s3_config(self) -> Dict[str, str]:
        """
        Retrieve the S3 configuration.

        Returns:
            Dict[str, str]: S3 configuration dictionary
        """
        return self.s3_config

    def get_serializable_config(self) -> Dict[str, Any]:
        """
        Get a serializable version of the entire configuration.

        This method is useful for saving or transmitting the configuration.

        Returns:
            Dict[str, Any]: Serializable configuration dictionary
        """
        return {
            'client_folder': self.client_folder,
            'client_config': self.client_config,
            's3_config': self.s3_config
        }





