import pandas as pd
import torch
from torch.utils.data import Dataset

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def transform_date_column_and_drop_it(df, date_column_name:str, remain_same = True):
    df = df.copy()
    if remain_same: 
        df.drop(date_column_name, axis = 1, inplace=True)
    else: 
        df['day'] = df[date_column_name].dt.day
        df['month'] = df[date_column_name].dt.month
        df['hour'] = df[date_column_name].dt.hour
        df['minute'] = df[date_column_name].dt.minute
        df['weekday'] = df[date_column_name].dt.dayofweek
        df.drop(date_column_name, axis = 1, inplace=True)
    return df

def split_dataset(df, feature_columns,  train_split_month=12, val_split_month=16, test_split_month=20): 
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    training_df = transform_date_column_and_drop_it(data[data['date'] < data['date'].min() + pd.DateOffset(months=train_split_month)],'date')
    val_df = transform_date_column_and_drop_it(data[data['date'] > data['date'].min() + pd.DateOffset(months=train_split_month)][data['date'] < data['date'].min() + pd.DateOffset(months=val_split_month)],'date')
    test_df = transform_date_column_and_drop_it(data[data['date'] > data['date'].min() + pd.DateOffset(months=val_split_month)][data['date'] < data['date'].min() + pd.DateOffset(months=test_split_month)],'date')
    return training_df[feature_columns], val_df[feature_columns], test_df[feature_columns]

class TimeSeriesDataset(Dataset):
    def __init__(self,df, target_column,feature_columns,future_steps, p_lag=0):
        self.df = df
        self.p_lag = p_lag
        self.len_df_minus_lag = len(self.df) - p_lag - future_steps
        self.target_column = target_column
        self.future_steps = future_steps
        self.feature_columns = feature_columns

    def __len__(self):
        return self.len_df_minus_lag

    def __getitem__(self, idx):
        input_p_lag = torch.tensor(self.df[self.feature_columns].iloc[(idx):(idx + self.p_lag),:].astype(float).to_numpy().transpose().reshape(1,-1), requires_grad=True)
        target = torch.tensor(self.df[self.target_column].iloc[(idx + self.p_lag): (idx + self.p_lag + self.future_steps),:].astype(float).to_numpy()).reshape(1,-1)
        return input_p_lag, target