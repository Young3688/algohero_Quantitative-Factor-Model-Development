import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def train_and_test(train_X, train_Y, test_X, test_Y):
    model = Ridge()
    parameters = {'alpha': np.logspace(-1, -4, 20)}

    grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
    grid.fit(train_X, train_Y)

    best_alpha = grid.best_params_['alpha']
    model = Ridge(alpha=best_alpha)
    model.fit(train_X, train_Y)
    
    Y_train_pred = model.predict(train_X)
    Y_test_pred = model.predict(test_X)
    
    lowest_loss = mean_squared_error(test_Y, Y_test_pred)
    r2_scores = r2_score(test_Y, Y_test_pred)

    coef_params = model.coef_

    return coef_params, lowest_loss, r2_scores, Y_train_pred, Y_test_pred

def model_construction_ridge(train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test):
    results = []

    for y_col in Y:
        # Y_suffix = y_col.split('_')[-1]
        train_Y = Y_train[[y_col]]
        test_Y = Y_test[[y_col]]
        # X_suffix = [x for x in X if x.endswith(Y_suffix)]
        # additional_x_cols = [x for x in X if not (x.endswith('h') or x.endswith('d'))]
        # x_cols = X_suffix  + [x for x in additional_x_cols if x not in X_suffix]
        
        for x_col in X:
            train_X = X_train[[x_col]]
            test_X = X_test[[x_col]]
            coef_params, lowest_loss, r2_scores, Y_train_pred, Y_test_pred = train_and_test(train_X, train_Y, test_X, test_Y)
            test_df[f'{x_col}___{y_col}_pred_rtn'] = Y_test_pred
            
            results.append((y_col, x_col, lowest_loss, r2_scores, coef_params))
            
    results_df = pd.DataFrame(results, columns=['Y', 'X', 'Lowest_Loss', 'R2_Scores', 'Coef_params'])
    result_df = results_df.sort_values(by='Lowest_Loss').head(5)
    
    return result_df, test_df