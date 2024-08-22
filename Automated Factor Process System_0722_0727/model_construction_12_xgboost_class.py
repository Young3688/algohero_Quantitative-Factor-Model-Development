import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

def transform_labels(y):
    # Transform -1, 0, 1 to 0, 1, 2
    return y + 1

def inverse_transform_labels(y):
    # Transform 0, 1, 2 back to -1, 0, 1
    return y - 1

def train_and_test(train_X, train_Y, test_X, test_Y):
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3)

    parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid = GridSearchCV(model, parameters, scoring='accuracy', cv=5)
    grid.fit(train_X, train_Y)

    best_params = grid.best_params_
    model = xgb.XGBClassifier(**best_params)
    model.fit(train_X, train_Y)

    # Predictions
    Y_train_pred = model.predict(train_X)
    Y_test_pred = model.predict(test_X)

    # Calculate metrics
    accuracy = accuracy_score(test_Y, Y_test_pred)
    precision = precision_score(test_Y, Y_test_pred, average='macro')
    recall = recall_score(test_Y, Y_test_pred, average='macro')
    f1 = f1_score(test_Y, Y_test_pred, average='macro')
    roc_auc = roc_auc_score(test_Y, model.predict_proba(test_X), average='macro', multi_class='ovr')

    # Return the results
    return accuracy, precision, recall, f1, roc_auc, Y_train_pred, Y_test_pred

def model_construction_xgboost(train_df, test_df, Y, X, Y_train, X_train, Y_test, X_test):
    results = []

    for y_col in Y:
        Y_suffix = y_col.split('_')[-1]
        train_Y = Y_train[[y_col]]
        test_Y = Y_test[[y_col]]
        
        # Transform labels
        train_Y_transformed = transform_labels(train_Y)
        test_Y_transformed = transform_labels(test_Y)
        
        X_suffix = [x for x in X if x.endswith(Y_suffix)]
        additional_x_cols = [x for x in X if not (x.endswith('h') or x.endswith('d'))]
        x_cols = X_suffix + [x for x in additional_x_cols if x not in X_suffix]
        
        for x_col in x_cols:
            train_X = X_train[[x_col]]
            test_X = X_test[[x_col]]
            accuracy, precision, recall, f1, roc_auc, Y_train_pred, Y_test_pred = train_and_test(train_X, train_Y_transformed, test_X, test_Y_transformed)
            
            # Inverse transform predictions
            Y_test_pred_original = inverse_transform_labels(Y_test_pred)
            test_df[f'{x_col}___{y_col}_pred_rtn'] = Y_test_pred_original
            
            results.append((y_col, x_col, accuracy, precision, recall, f1, roc_auc))
            
    results_df = pd.DataFrame(results, columns=['Y', 'X', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC'])
    result_df = results_df.sort_values(by='Accuracy', ascending=False).head(5)
    
    return result_df, test_df