import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np

def fill_date_gaps_with_covariates_test_only(
    df,
    item_covars=None,
    time_covars=None,
    item_time_covars=None,
    fill_value=0,
    fill_value_covars=None,
    date_col="date",
    remove_weekends=False,
    fill_covars_with_frequent=False,
):
    """
    Fill the gaps between observations in the dataframe for each unique_id,
    merging item and time covariates based on unique_id and date respectively.

    Args:
        df (pd.DataFrame): Input dataframe
        item_covars (list): List of item-level covariates to merge on unique_id
        time_covars (list): List of time-based covariates to merge on date
        item_time_covars (list): List of item-time covariates to merge on unique_id and date
        fill_value (int, float, str): Value to use for filling missing values in the y column
        fill_value_covars (int, float, str): Value to use for filling missing values in the covariates columns
        date_col (str): Name of the date column
        remove_weekends (bool): Whether to remove weekends from the date range
        fill_covars_with_frequent (bool): Whether to fill covariate NaNs with the most frequent value. Default: False

    Returns:
        pd.DataFrame: Dataframe with filled date gaps and merged covariates
    """

    # Initializing some values
    id_col = "unique_id"

    # Take the full date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    if remove_weekends:
        date_range = pd.date_range(start=min_date, end=max_date, freq='B')  # 'B' frequency excludes weekends
    else:
        date_range = pd.date_range(start=min_date, end=max_date)
    unique_ids = df[id_col].unique()
    
    all_combinations = pd.MultiIndex.from_product(
        [unique_ids, date_range], names=[id_col, date_col]
    ).to_frame(index=False)

    # Prepare covariates
    all_covars = (item_covars or []) + (time_covars or []) + (item_time_covars or [])
    covar_columns = [id_col, date_col] + all_covars
    covar_df = df[covar_columns].drop_duplicates()

    # Merge all at once
    uid_df = all_combinations.merge(covar_df, on=[id_col, date_col], how="left")

    # Merge y values
    y_df = df[[id_col, date_col, "y"]].drop_duplicates()
    uid_df = uid_df.merge(y_df, on=[id_col, date_col], how="left")

    # Fill missing y values
    uid_df["y"] = uid_df["y"].fillna(fill_value)

    if fill_covars_with_frequent:
        # Precompute frequent values for item covariates (static features)
        frequent_item_values = {}
        for covar in (item_covars or []):
            if covar in uid_df.columns:
                frequent_item_values[covar] = uid_df.groupby(id_col)[covar].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)

        # Precompute frequent values for item-time covariates
        frequent_item_time_values = {}
        for covar in (item_time_covars or []):
            if covar in uid_df.columns:
                frequent_item_time_values[covar] = uid_df.groupby(date_col)[covar].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)

        # Fill NaNs for item covariates (static features)
        for covar, values in frequent_item_values.items():
            mask = uid_df[covar].isna()
            uid_df.loc[mask, covar] = uid_df.loc[mask, id_col].map(values)

        # Fill NaNs for item-time covariates
        for covar, values in frequent_item_time_values.items():
            mask = uid_df[covar].isna()
            uid_df.loc[mask, covar] = uid_df.loc[mask, date_col].map(values)

        # Fill NaNs for time covariates (if any)
        for covar in (time_covars or []):
            if covar in uid_df.columns:
                uid_df[covar].fillna(uid_df[covar].mode().iloc[0], inplace=True)

    elif fill_value_covars is not None:
        uid_df[all_covars] = uid_df[all_covars].fillna(fill_value_covars)

    # Drop duplicates
    return uid_df


def check_covars(covars, merge):
    """
    Checks if the covars and merge are in the right format.
    """

    if isinstance(covars, str):
        covars = [covars]

    if isinstance(covars, list):
        # check if merge if a string
        if isinstance(merge, str):
            # convert to list with the same length as covars
            merge = [merge] * len(covars)
            # raise warning here
            print(
                "Warning: Different number of merge columns and covariates. Using the first merge column for all covariates."  # noqa 501
            )
        # if its a list with not the same length raise an error
        elif len(merge) != len(covars):
            raise ValueError(
                "The merge column and covariates should have the same length."
            )

    return covars, merge


def pivoted_df(df, target_frequency=None, agg_func=None, fill_values=True, fillna=True):
    """
    Converts a transaction df to a pivoted df.
    Each row is a unique id and columns are the dates.
    Missing values are filled with zeros by default.
    Time series can be resampled to different frequencies.

    Args:
        df (pd.DataFrame): A transaction DataFrame with columns 'date', 'y', and 'unique_id'.
        target_frequency (str): Target frequency for resampling. Ex: 'D' for daily, 'W' for weekly.
        agg_func (str): The aggregation function. Options: 'sum', 'constant', None. Default: None.
        fill_values (bool): Whether or not to fill missing values with zeros. Default: True.
        fillna (bool): Whether or not to fill missing values with zeros. Default: True.

    Returns:
        pd.DataFrame: A pivoted DataFrame.

    Examples:
        >>> df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-01', '2022-01-02',
                            '2022-01-03'],
        ...                    'y': [1, 2, 3, 4, 5, 6],
        ...                    'unique_id': ['A', 'A', 'A', 'B', 'B', 'B']})
        >>> pivoted_df(df, 'D', 'sum')
                    2022-01-01  2022-01-02  2022-01-03
        unique_id
        A                   1           2           3
        B                   4           5           6
    """

    # Ensure dates are on the right formatI
    df["date"] = pd.to_datetime(df["date"])

    # Pivots on the original frequency
    pivot_df = pd.pivot_table(
        df, index="unique_id", columns="date", values="y", aggfunc="first"
    )

    # Drop values with full nans
    pivot_df = pivot_df.dropna(axis=0, how="all")

    if target_frequency is not None:
        # Resamples with the given function
        # for sales data
        if agg_func == "sum":
            pivot_df = pivot_df.resample(target_frequency, axis=1).sum()
        # for stock data
        elif agg_func == "constant":
            pivot_df = pivot_df.resample(target_frequency, axis=1).last()

    # Fills missing values
    if fill_values:
        pivot_df = pivot_df.reindex(
            columns=pd.date_range(
                pivot_df.columns.min(),
                pivot_df.columns.max(),
                freq=target_frequency,
            )
        )
    if fillna:
        pivot_df = pivot_df.fillna(0)

    return pivot_df



def get_numeric_frequency(freq):
    """
    Return the frequency of a time series in numeric format.

    The function returns the frequency of a time series in numeric format. This is useful when working with
    forecasting libraries that require the frequency to be a number instead of a string.

    If frequency has multiple seasonalities, for example Daily and Hourly, returns a list with all periods.

    Args:
        freq (str): A string specifying the frequency of the time series.
        Valid values are:
        'Y' (yearly), 'A' (annually), 'Q' (quarterly), 'M' (monthly), 'W' (weekly), 'D' (daily), or 'H' (hourly).

    Returns:
        int: The frequency of the time series in numeric format if frequency has only one seasonalities.
        list: A list with all periods if frequency has multiple seasonalities.

    References:
        - https://otexts.com/fpp3/tsibbles.html

    Example:
        >>> get_numeric_frequency('M')
        1

        >>> get_numeric_frequency('W')
        13

        >>> get_numeric_frequency('D')
        365
    """

    keys = ["Y", "A", "Q", "M", "W", "D", "H", 'B']
    vals = [1, 1, 4, 12, 52, [7, 30, 364], [24, 168, 720, 8760], 5]

    freq_dictionary = dict(zip(keys, vals))

    # Getting the period and the frequency
    period = to_offset(freq).n

    # Taking the first letter of the frequency in case we have MS for month start etc
    freq = to_offset(freq).name[0]

    # Initializing the dictionary
    numeric_freq = freq_dictionary[freq]

    # Dividing with the period:
    # For example if I have a 2M frequency:
    # Then instead of 12 months we have 6 examina
    numeric_freq = (
        int(freq_dictionary[freq] / period)
        if isinstance(numeric_freq, int)
        else [int(i / period) for i in numeric_freq]
    )

    return numeric_freq

def statsforecast_forecast_format(
    df, format="transaction", fill_missing=False, fill_value="nan"
):
    """
    Converts a dataframe to the format required for forecasting with statsforecast.

    Args:
        df : pd.DataFrame
            The input data.
        format : str, default='transaction'
            The format of the input data. Can be 'transaction' or 'pivotted'.
        fill_missing : bool, default=False
            Whether to fill missing dates with NaN values. If True, the 'fill_value' argument
            must be specified.
        fill_value : str, int, float, default='nan'
            The value to use for filling missing dates. Default is 'nan'.

    Returns:
        df : pd.DataFrame
            The formatted dataframe.

    """

    # if we have transaction
    if format == "transaction":
        # just rename the date column to ds
        df = df.rename(columns={"date": "ds"})

        # fill missing dates if specified
        if fill_missing:
            df = fill_missing_dates(df, fill_value=fill_value)

    elif format == "pivotted":
        # if we have pivotted
        # we need to convert it to transaction
        df = transaction_df(df, drop_zeros=False)
        # and rename the date column to ds
        df = df.rename(columns={"date": "ds"})
    else:
        raise ValueError(
            "Provide the dataframe either in pivoted or transactional format."
        )

    # Return
    return df


def transaction_df(df, drop_zeros=False):
    """
    Converts a pivoted df to a transaction df. A transaction df has 3 columns:
    - unique_id: Sales location of each time series.
    - date: The date.
    - y: The value for the time series.

    Args:
        df (pd.DataFrame): The pivoted DataFrame with time series as rows and dates as columns.
        drop_zeros (bool): Whether or not to drop periods with zero sales. Default: False.

    Returns:
        pd.DataFrame: A transaction DataFrame.

    Examples:
        >>> df = pd.DataFrame({'unique_id': ['A', 'A', 'B', 'B'], '2022-01-01': [1, 2, 0, 4],
                '2022-01-02': [0, 5, 6, 0]})
        >>> transaction_df(df)
        unique_id        date  y
        0         A  2022-01-01  1
        1         A  2022-01-01  2
        2         B  2022-01-02  6
        3         B  2022-01-01  4
        >>> transaction_df(df, drop_zeros=True)
        unique_id
    """

    # resets the index
    trans_df = df.reset_index(names="unique_id")

    # Melts
    trans_df = pd.melt(trans_df, id_vars="unique_id", value_name="y", var_name="date")

    # Filters zeros if keep_zeros is set to True
    if drop_zeros:
        trans_df = trans_df[trans_df["y"] != 0]

    return trans_df


def fill_missing_dates(df, fill_value="nan"):
    """
    Fills missing dates due to no sales.

    Args:
        df : pd.DataFrame
            The input data, expected to have at least 'ds' (date), 'y' (target variable),
            and 'unique_id' columns.
        fill_value: str, int
            The value to use for filling missing dates. Default is 'nan'.

    Returns:
        df : pd.DataFrame
            The formatted DataFrame with a continuous date range for each 'unique_id',
            filling missing dates with NaN values for 'y', and ensuring that the 'ds' column is
            of datetime type. The returned DataFrame is sorted by 'unique_id' and 'ds'.
    """

    # Identify the full date range in the dataset
    min_date, max_date = df["ds"].min(), df["ds"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    # Create a MultiIndex with all combinations of 'unique_id' and 'all_dates'
    unique_ids = df["unique_id"].unique()
    multi_index = pd.MultiIndex.from_product(
        [unique_ids, all_dates], names=["unique_id", "ds"]
    )

    # Reindex the DataFrame to include missing dates
    df_reindexed = df.set_index(["unique_id", "ds"]).reindex(multi_index).reset_index()

    # Sort by 'unique_id' and 'ds'
    df_reindexed = df_reindexed.sort_values(by=["unique_id", "ds"])

    if fill_value != "nan":
        df_reindexed = df_reindexed.fillna(fill_value)

    return df_reindexed