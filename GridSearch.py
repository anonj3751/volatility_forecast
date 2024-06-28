import time
import pandas as pd
import xgboost_fit
import itertools
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


def fit_XGBoost_model(dtrain, dval, depth, rate_drop, ratio, method):
    params = {
        "tree_method": "hist", 'booster': 'dart',  # 'booster': 'gbtree', #'n_estimators': 500, # general
        'max_depth': depth, "rate_drop": rate_drop,  # 'subsample': 0.5 #'colsample_bytree': 1, # complexity
        # 'lambda': 4, 'alpha': 4 # L2 & L1 regularisation
    }
    if method == 'CLASSIFIER':
        params["objective"] = "binary:logistic"
        # adjust for class imbalance
        params['scale_pos_weight'] = ratio
        # train xgboost
        model = xgboost_fit.xgboost_train_classification(dtrain=dtrain, params=params,
                                                         num_boost_round=30, dval=dval, verbose_eval=10)
    else:
        params["objective"] = "reg:squarederror"
        model = xgboost_fit.xgboost_train_regression(dtrain=dtrain, params=params,
                                                     num_boost_round=30, dval=dval, verbose_eval=10)
    return model


def perform_grid_search_crossval(set_number, depths, rates_drop, thresholds, method,
                                 objective, remove_curve_outliers, n_splits=5, basepath='GRIDSEARCH/RESULTS/'):
    start_gridsearch = time.time()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    assert method in {'CLASSIFIER', 'REGRESSION'}
    param_list_training = [depths, rates_drop]

    depths_list = []
    rates_list = []
    thresholds_list = []
    # for CLASSIFIER
    f1_train_list = []
    f1_val_list = []
    f1_test_list = []
    # for REGRESSION
    mse_train_list = []
    mse_val_list = []
    mse_test_list = []

    perform_regression = (method == 'REGRESSION')
    _, (X, y), (X_test, y_test) = xgboost_fit.prepare_data(set_number,
                                                           perform_regression, objective,
                                                           include_val=False,
                                                           remove_curve_outliers=remove_curve_outliers)
    # Fix test set
    my_dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)
    my_d_all_train = xgb.DMatrix(X, y, enable_categorical=True)

    param_combinations_training = itertools.product(*param_list_training)
    for (depth, rate_drop) in param_combinations_training:
        split_f1_train, split_f1_val, split_f1_test = [], [], []
        split_mse_train, split_mse_val, split_mse_test  = [], [], []
        # perform CV-split and fitting for each split
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            my_dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
            my_dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)

            ratio = 1
            if method == 'CLASSIFIER':
                ratio = float(len(y_train[y_train == 0])) / len(y_train[y_train == 1])
            model = fit_XGBoost_model(my_dtrain, my_dval, depth, rate_drop, ratio, method)

            # make prediction on test and train sets
            preds_train = model.predict(my_dtrain)
            preds_val = model.predict(my_dval)

            if method == 'CLASSIFIER':
                # test for different thresholds
                array_1 = np.zeros( len(thresholds) )
                array_2 = np.zeros( len(thresholds) )
                for threshold_idx, threshold in enumerate(thresholds):
                    array_1[threshold_idx] = f1_score(y_train, preds_train > threshold)
                    array_2[threshold_idx] = f1_score(y_val, preds_val > threshold)
                split_f1_train.append(array_1)
                split_f1_val.append(array_2)
            else:
                split_mse_train.append(mean_squared_error(y_train, preds_train, squared=True))
                split_mse_val.append(mean_squared_error(y_val, preds_val, squared=True))

        # We also collect data on the test-set performance (NOT for model selection!)
        # But doing so, we don't have to retrain later
        # To collect the data, rerun the training on the whole training set (without split)
        ratio = 1
        if method == 'CLASSIFIER':
            ratio = float(len(y[y == 0])) / len(y[y == 1])
        model = fit_XGBoost_model(my_d_all_train, my_d_all_train, depth, rate_drop, ratio, method)
        preds_test = model.predict(my_dtest)

        if method == 'CLASSIFIER':
            # test for different thresholds
            array_3 = np.zeros(len(thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                array_3[threshold_idx] = f1_score(y_test, preds_test > threshold)
            split_f1_test.append(array_3)
        else:
            split_mse_test.append(mean_squared_error(y_test, preds_test, squared=True))

        # Store the results for the given gridpoint
        if method=='CLASSIFIER':
            f1_train = np.mean( np.array(split_f1_train), axis=0 )
            f1_val = np.mean(np.array(split_f1_val), axis=0)
            f1_test = np.array(split_f1_val)[0]         # the array is one dimensional
            for threshold_idx, threshold in enumerate(thresholds):
                print('#################')
                print(f'Scores for rate_drop={rate_drop}, depth={depth}, threshold={threshold}')
                print(f"F1 score of the base model (Train set): {f1_train[threshold_idx]:.3f}")
                print(f"F1 score of the base model (Validation set): {f1_val[threshold_idx]:.3f}")
                print(f"F1 score of the base model (Test set): {f1_test[threshold_idx]:.3f}")
                # add scores to dataset
                depths_list.append(depth)
                rates_list.append(rate_drop)
                thresholds_list.append(threshold)
                f1_train_list.append(f1_train[threshold_idx])
                f1_val_list.append(f1_val[threshold_idx])
                f1_test_list.append(f1_test[threshold_idx])
        else:
            mse_train = np.mean( np.array(split_mse_train) )
            mse_val = np.mean( np.array(split_mse_val) )
            mse_test = np.array(split_mse_test)[0]        # the array is one dimensional
            print('#################')
            print(f'Scores for rate_drop={rate_drop}, depth={depth}')
            print(f"MSE of the base model (Train set): {mse_train:.3f}")
            print(f"MSE of the base model (Validation set): {mse_val:.3f}")
            print(f"MSE of the base model (Test set): {mse_test:.3f}")
            # add MSEs to dataset
            depths_list.append(depth)
            rates_list.append(rate_drop)
            mse_train_list.append(mse_train)
            mse_val_list.append(mse_val)
            mse_test_list.append(mse_test)

    # store results
    if method == 'CLASSIFIER':
        results_df = pd.DataFrame({'depth': depths_list, 'rate': rates_list, 'threshold': thresholds_list,
                                   'f1_train': f1_train_list,
                                   'f1_val': f1_val_list,
                                   'f1_test': f1_test_list}
                                  )
    else:
        results_df = pd.DataFrame({'depth': depths_list, 'rate': rates_list,
                                   'mse_train': mse_train_list,
                                   'mse_val': mse_val_list,
                                   'mse_test': mse_test_list })
    results_df.to_pickle(basepath + f'{method}_ob_{objective}_set{set_number}.pkl')
    end_gridsearch = time.time()
    print(f'time for gridsearch: {end_gridsearch - start_gridsearch:.3f}')


if __name__ == '__main__':
    my_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    my_rates_drop = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    my_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    my_set_number = 1

    print(f'HANDLING SET {my_set_number}...')
    print("##########################################")
    print("##########################################")
    print('START REGRESSION ')
    print("##########################################")
    print("##########################################")
    # perform Gridsearch for regression
    perform_grid_search_crossval(my_set_number, my_depths, my_rates_drop, my_thresholds,
                                 method='REGRESSION', objective=1, remove_curve_outliers=False,
                                 basepath='')

    print("##########################################")
    print("##########################################")
    print('START CLASSIFICATION -- OBJECTIVE 1 ')
    print("##########################################")
    print("##########################################")
    # perform Gridsearch for classification -- Objective 1
    perform_grid_search_crossval(my_set_number, my_depths, my_rates_drop, my_thresholds,
                                 method='CLASSIFIER', objective=1, remove_curve_outliers=False,
                                 basepath='')

    print("##########################################")
    print("##########################################")
    print('START CLASSIFICATION -- OBJECTIVE 2 ')
    print("##########################################")
    print("##########################################")
    # perform Gridsearch for classification -- Objective 2
    perform_grid_search_crossval(my_set_number, my_depths, my_rates_drop, my_thresholds,
                                 method='CLASSIFIER', objective=2, remove_curve_outliers=False,
                                 basepath='')


