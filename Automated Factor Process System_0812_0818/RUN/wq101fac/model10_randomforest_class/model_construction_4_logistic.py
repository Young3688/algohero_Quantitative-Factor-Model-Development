# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_test(train_X, train_Y, test_X, test_Y):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(train_X, train_Y.values.ravel())
    Y_train_pred = model.predict(train_X)
    Y_test_pred = model.predict(test_X)
    acc = accuracy_score(test_Y, Y_test_pred)
    prec = precision_score(test_Y, Y_test_pred, average='macro') 
    rec = recall_score(test_Y, Y_test_pred, average='macro')
    f1 = f1_score(test_Y, Y_test_pred, average='macro')
    return acc, prec, rec, f1, Y_train_pred, Y_test_pred

def model_construction_logistic(train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test):
    results = []
    for y_col in Y:
        train_Y = Y_train[[y_col]]
        test_Y = Y_test[[y_col]]
        # Y_suffix = y_col.split('_')[-1]
        # X_suffix = [x for x in X if x.endswith(Y_suffix)]
        # additional_x_cols = [x for x in X if not x.endswith('h') and not x.endswith('d')]
        # x_cols = X_suffix + additional_x_cols
        for x_col in X:
            train_X = X_train[[x_col]]
            test_X = X_test[[x_col]]
            acc, prec, rec, f1, Y_train_pred, Y_test_pred = train_and_test(train_X, train_Y, test_X, test_Y)
            test_df[f'{x_col}___{y_col}_pred_rtn'] = Y_test_pred         
            results.append((y_col, x_col, acc, prec, rec, f1))
    results_df = pd.DataFrame(results, columns=['Y', 'X', 'Accuracy', 'Precision', 'Recall', 'F1'])
    result_df = results_df.sort_values(by='Accuracy', ascending=False).head(5)
    
    return result_df, test_df

# def train_and_test(train_X, train_Y, test_X, test_Y):


#     model = LogisticRegression()

#     model.fit(train_X, train_Y.values.ravel())

#     Y_train_pred = model.predict(train_X)
#     Y_test_pred = model.predict(test_X)
    
#     Y_test_proba = model.predict_proba(test_X)[:, 1]

#     acc = accuracy_score(test_Y, Y_test_pred)
#     prec = precision_score(test_Y, Y_test_pred)
#     rec = recall_score(test_Y, Y_test_pred)
#     f1 = f1_score(test_Y, Y_test_pred)
#     auc = roc_auc_score(test_Y, Y_test_proba)

#     return acc, prec, rec, f1, auc, Y_train_pred, Y_test_pred

# def model_construction_logistic(train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test):
#     results = []

#     for y_col in Y:
#         train_Y = (Y_train[[y_col]] > 0).astype(int)
#         test_Y = (Y_test[[y_col]] > 0).astype(int)
#         Y_suffix = y_col.split('_')[-1]
#         X_suffix = [x for x in X if x.endswith(Y_suffix)]
#         additional_x_cols = [x for x in X if not x.endswith('h') and not x.endswith('d')]
#         x_cols = X_suffix + additional_x_cols
        
#         for x_col in x_cols:
#             train_X = X_train[[x_col]]
#             test_X = X_test[[x_col]]
#             acc, prec, rec, f1, auc, Y_train_pred, Y_test_pred = train_and_test(train_X, train_Y, test_X, test_Y)
#             test_df[f'{x_col}___{y_col}_pred_rtn'] = Y_test_pred 
            
#             results.append((y_col, x_col, acc, prec, rec, f1, auc))
    
#     results_df = pd.DataFrame(results, columns=['Y', 'X', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
#     result_df = results_df.sort_values(by='Accuracy', ascending=False).head(5)
    
#     return result_df, test_df