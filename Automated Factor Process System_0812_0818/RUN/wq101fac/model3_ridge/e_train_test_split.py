import pandas as pd
import numpy as np

# def train_test_split(model_df, train_start, train_end, test_start, test_end):
#     train_df = model_df[(model_df.index >= train_start) & (model_df.index < train_end)]
#     test_df = model_df[(model_df.index >= test_start) & (model_df.index < test_end)]

#     Y = model_df.filter(like='return').columns.tolist()
#     max_return_index = max(model_df.columns.get_loc(col) for col in Y)
#     X = model_df.columns[max_return_index + 1:].tolist()

#     Y_train = train_df[Y]
#     X_train = train_df[X]
    
#     Y_test = test_df[Y]
#     X_test = test_df[X]
    
#     return train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test


def train_test_split(model_df, train_start, train_end, test_start, test_end):
    train_df = model_df[(model_df.index >= train_start) & (model_df.index < train_end)]
    test_df = model_df[(model_df.index >= test_start) & (model_df.index < test_end)]

    Y = model_df.filter(like='return').columns.tolist()
    max_return_index = max(model_df.columns.get_loc(col) for col in Y)
    X = model_df.columns[max_return_index + 1:].tolist()

    Y_train = train_df[Y].shift(-1).dropna()
    X_train = train_df[X][:-1]
    
    Y_test = test_df[Y].shift(-1).dropna()
    X_test = test_df[X][:-1]
    
    train_df = train_df[:-1]
    test_df = test_df[:-1]
    
    return train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test

