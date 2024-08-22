import numpy as np
import pandas as pd

def handle_return(df, column_names):
    def handle_missing_values(column):
        if column.isnull().any():
            if column.skew() > 0.5 or column.skew() < -0.5:
                fill_value = column.median()
            else:
                fill_value = column.mean()
            column.fillna(fill_value, inplace=True)
        return column
    for col in column_names:
        if df[col].dtype in ['float64', 'int64']: 
            df[col] = handle_missing_values(df[col])
    return df

def remove_outliers(column):
    D_M = column.median()
    D_M1 = (column - D_M).abs().median()
    upper_limit = D_M + 5 * D_M1
    lower_limit = D_M - 5 * D_M1
    return np.clip(column, lower_limit, upper_limit)

def handle_missing_values(column):
    if column.isnull().any():
        if column.skew() > 0.5 or column.skew() < -0.5:
            fill_value = column.median()
        else:
            fill_value = column.mean()
        column.fillna(fill_value, inplace=True)
    return column

def standardize_data(column):
    mean = column.mean()
    std_dev = column.std()
    return (column - mean) / std_dev

def normalize_data(column):
    X_min = column.min()
    X_max = column.max()
    return (column - X_min) / (X_max - X_min) if (X_max - X_min) != 0 else 0

def feature_preprocessing(df, column_names, method=None):
    for col in column_names:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = remove_outliers(df[col])
            df[col] = handle_missing_values(df[col])
            if method == 'standardize':
                df[col] = standardize_data(df[col])
            elif method == 'normalize':
                df[col] = normalize_data(df[col])
            else:
                raise ValueError("Method must be 'standardize' or 'normalize'")
    
    missing_values = df.isna().any().any()
    print("Are there any missing values in the DataFrame? ", missing_values)
    
    return df