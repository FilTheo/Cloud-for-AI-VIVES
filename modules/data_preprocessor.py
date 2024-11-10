from logs.logging_config import setup_logger
import pandas as pd
from modules.utils.utils import fill_date_gaps_with_covariates_test_only
#from modules.client_preprocess import preprocess_pipeline

# Set up logger
logger = setup_logger(__name__, 'data_preprocessor.log')

class DataPreprocessor:
    """
    Takes the raw data from the Extractor class and transforms the data to the universal format
    for forecasting.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataPreprocessor with raw data.

        Args:
            data (pd.DataFrame): The raw data extracted from extractor
        """
        logger.info("Initializing DataPreprocessor")
        self.data = data
        logger.info(f"DataPreprocessor initialized with data shape: {self.data.shape}")


    def convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the date column to a pandas datetime object.

        Args:
            df (pd.DataFrame): The raw data

        Returns:
            pd.DataFrame: The data with datetime column converted to pandas datetime object
        """
        logger.info("Converting date column to datetime")
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
            logger.info("Date conversion successful")
        except Exception as e:
            logger.error(f"Error converting date column: {str(e)}")
        return df

    def prepare_covariates(
        self,
        df: pd.DataFrame,
        univariate_columns: list = ["date", "unique_id", "y"],
        fill_na: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare the covariates for the model.

        Args:
            df (pd.DataFrame): The raw data
            univariate_columns (list): Columns to exclude from covariate processing
            fill_na (bool): Whether to fill NA values in covariates

        Returns:
            pd.DataFrame: The data with covariates prepared
        """
        logger.info("Preparing covariates")

        # Extract the covariates by excluding univariate columns
        covariate_columns = [col for col in df.columns if col not in univariate_columns]
        logger.info(f"Covariate columns: {covariate_columns}")

        # Convert non-numeric columns to category codes
        for col in covariate_columns:
            if df[col].dtype not in ["float64", "int64"]:
                logger.info(f"Converting column {col} to category codes")
                # if we have numeric values convert to int
                if col == 'Holiday':
                    df[col] = df[col].astype(float)
                df[col] = df[col].astype("category")

        logger.info("Covariate preparation completed")
        return df

    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data types of the columns to the appropriate types.

        Args:
            df (pd.DataFrame): The raw data

        Returns:
            pd.DataFrame: The data with the columns converted to the appropriate types
        """
        logger.info("Converting column types")
        # TODO: Implement type conversion logic
        logger.info("Type conversion completed")
        return df

    def fill_date_range(self, df: pd.DataFrame, remove_weekends: bool = False,
                        fill_covars_with_frequent: bool = True,
                        fill_value: int = 0,
                        fill_value_covars: int = None,
                        item_covars: list = None,
                        time_covars: list = None,
                        item_time_covars: list = None,
                        date_col: str = "date",
                        ) -> pd.DataFrame:
        """
        Fill the date range so that there are no missing dates in the data.

        Args:
            df (pd.DataFrame): The raw data
            remove_weekends (bool): Whether to remove weekends from the date range

        Returns:
            pd.DataFrame: The data with the date range filled
        """
        logger.info("Filling date range")
        df = fill_date_gaps_with_covariates_test_only(df, remove_weekends=remove_weekends,
                                                            fill_covars_with_frequent=fill_covars_with_frequent,
                                                            fill_value=fill_value,
                                                            fill_value_covars=fill_value_covars,
                                                            item_covars=item_covars,
                                                            time_covars=time_covars,
                                                            item_time_covars=item_time_covars,
                                                            date_col=date_col,
                                                            )
        logger.info(f"Date range filled. New shape: {df.shape}")
        return df
    

    def preprocess_new_data(self, client_name: str, new_file: bool = False) -> pd.DataFrame:
        """
        Preprocess the new data for the client.

        Args:
            client_name (str): The name of the client

        Returns:
            pd.DataFrame: The preprocessed data
        """

        logger.info("Starting preprocessing of the latest file")

        if client_name == 'client_1':
            df = preprocess_pipeline(self.data, new_file = new_file)
        elif client_name == 'm5':
            df = self.data
        else:
            raise ValueError(f"Client name {client_name} not recognized")
        
        return df

    def preprocess(self, remove_weekends: bool = False,
                   fill_covars_with_frequent: bool = True,
                   fill_value: int = 0,
                   fill_value_covars: int = None,
                   item_covars: list = None,
                   time_covars: list = None,
                   item_time_covars: list = None,
                   date_col: str = "date",
                   filter_dates: bool = False,
                   threshold: int = 180,
                   ) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline: function_1, function_2, function_3 etc.
        When I focus on Arthurs I will run these functions.

        Args:
            remove_weekends (bool): Whether to remove weekends from the date range

        Returns:
            pd.DataFrame: The fully preprocessed data

        """

        logger.info("Starting preprocessing pipeline")


        # Convert datetime
        logger.info("Step 1: Converting datetime")
        proccessed_data = self.convert_datetime(self.data)

        # Fill date range
        logger.info("Step 2: Filling date range")
        proccessed_data = self.fill_date_range(proccessed_data, remove_weekends=remove_weekends,
                                                fill_covars_with_frequent=fill_covars_with_frequent,
                                                fill_value=fill_value,
                                                fill_value_covars=fill_value_covars,
                                                item_covars=item_covars,
                                                time_covars=time_covars,
                                                item_time_covars=item_time_covars,
                                                date_col=date_col,
                                                )

                                                 

        # Convert types
        logger.info("Step 3: Converting types")
        proccessed_data = self.convert_types(proccessed_data)

        # Convert covariates
        logger.info("Step 4: Preparing covariates")
        proccessed_data = self.prepare_covariates(proccessed_data)

        logger.info("Preprocessing pipeline completed")
        logger.info(f"Final preprocessed data shape: {proccessed_data.shape}")


        if filter_dates:
            logger.info("Filtering dates")
            dates = proccessed_data['date'].unique()
            dates_half = dates[threshold:]
            proccessed_data = proccessed_data[proccessed_data['date'].isin(dates_half)]
            
        return proccessed_data
