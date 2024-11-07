from sats.local_preprocess.lags import create_lags
from sats.TS.ts_class import *
import pandas as pd

df = pd.read_csv('air.csv').set_index('Month')
ts_df = ts(df)

lagged_ts = create_lags(ts_df, lags = 'default', remove_zeros= True)
