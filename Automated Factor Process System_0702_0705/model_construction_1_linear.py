import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_and_test(train_X, train_Y, test_X, test_Y):
    
    model = LinearRegression()
    
    train_losses, test_losses = [], []
    train_scores, test_scores = [], []

    coef_params = None
    lowest_loss = np.inf

    model.fit(train_X, train_Y)
        
    Y_train_pred = model.predict(train_X)
    Y_test_pred = model.predict(test_X)
    
    train_loss = mean_squared_error(train_Y, Y_train_pred)
    test_loss = mean_squared_error(test_Y, Y_test_pred)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
        
    train_score = r2_score(train_Y, Y_train_pred)
    test_score = r2_score(test_Y, Y_test_pred)
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    lowest_loss = test_loss
    coef_params = model.coef_
    r2_scores = test_scores

    return coef_params, lowest_loss, r2_scores, Y_train_pred, Y_test_pred

def model_construction_linear(train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test):
    results = []

    for y_col in Y:
        Y_suffix = y_col.split('_')[-1]
        train_Y = Y_train[[y_col]]
        test_Y = Y_test[[y_col]]
        X_suffix = [x for x in X if x.endswith(Y_suffix)]
        additional_x_cols = [x for x in X if not (x.endswith('h') or x.endswith('d'))] # maybe change
        x_cols = X_suffix  + [x for x in additional_x_cols if x not in X_suffix]
        for x_col in x_cols:
            train_X = X_train[[x_col]]
            test_X = X_test[[x_col]]
            coef_params, lowest_loss, r2_scores, Y_train_pred, Y_test_pred = train_and_test(train_X, train_Y, test_X, test_Y)
            test_df[f'{x_col}___{y_col}_pred_rtn'] = Y_test_pred
            
            results.append((y_col, x_col, lowest_loss, r2_scores, coef_params))
            
    results_df = pd.DataFrame(results, columns=['Y', 'X', 'Lowest_Loss', 'R2_Scores', 'Coef_params'])
    result_df = results_df.sort_values(by='Lowest_Loss').head(5)
    
    return result_df, test_df
