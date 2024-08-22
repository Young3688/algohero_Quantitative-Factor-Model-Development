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

def is_binary(column):
    unique_values = column.dropna().unique()
    return len(unique_values) == 2

def handle_binary(column):
    if column.isnull().any():
        fill_value = column.mode()[0]
        column.fillna(fill_value, inplace=True)
    return column

def remove_outliers(column):
    D_M = column.median()
    D_M1 = (column - D_M).abs().median()
    if D_M == 0 or D_M1 == 0:
        return column
    else:
        upper_limit = D_M + 5 * D_M1
        lower_limit = D_M - 5 * D_M1
        clipped_column = np.clip(column, lower_limit, upper_limit)
        return clipped_column

# def handle_missing_values(column):
#     if column.isnull().any() == True:
#         if column.skew() > 0.5 or column.skew() < -0.5:
#             fill_value = column.median()
#         else:
#             fill_value = column.mean()
#         column.fillna(fill_value, inplace=True)
#         return column
#     else:
#         return column

# def handle_missing_values(column):
#     return column.dropna()
    
def standardize_data(column):
    mean = column.mean()
    std_dev = column.std()
    if std_dev == 0:
        return column
    else:
        standardized_column = (column - mean) / std_dev
        return standardized_column

def normalize_data(column):
    X_min = column.min()
    X_max = column.max()
    if X_max == X_min or X_max == 0 or X_min == 0: 
        return column
    else:
        normalized_column = (column - X_min) / (X_max - X_min)
        return normalized_column

def feature_preprocessing(df, column_names, method=None):
    df = df.dropna()
    for col in column_names:
        if df[col].dtype in ['float64', 'int64']:
            if is_binary(df[col]):
                df[col] = handle_binary(df[col])
            else:
                df[col] = remove_outliers(df[col])
                if method == 'standardize':
                    df[col] = standardize_data(df[col])
                elif method == 'normalize':
                    df[col] = normalize_data(df[col])
                else:
                    raise ValueError("Method must be 'standardize' or 'normalize'")
    
    missing_values = df.isna().any().any()
    print("Are there any missing values in the DataFrame? ", missing_values)
    return df