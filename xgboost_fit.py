import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
from sklearn.metrics import f1_score

from helper import swap_levels, unpivot
from feature_sets import get_set
from feature_sets import get_categories_columns


# assumes target column is called 'target'
def get_variables(my_df):
    features = set(my_df.columns)
    features -= { 'date', 'ticker', 'target' }
    GICS_onehotencodings = {col for col in features if 'GICS_' in col}
    features -= GICS_onehotencodings    # remove them
    X, y = my_df[ list(features) ], my_df[ 'target' ]
    return X, y


def xgboost_train_regression(dtrain, params, num_boost_round=500, verbose_eval=50, dval=None):
    evals = [(dtrain, "train")]

    # Add validation dataset to the evaluations if provided
    if dval:
        evals.append((dval, "validation"))

    # Train the XGBoost model
    model = xgb.train(
       params=params,
       dtrain=dtrain,
       num_boost_round=num_boost_round,
       evals=evals,
       verbose_eval=verbose_eval,
    )
    return model


def xgboost_train_classification(dtrain, params, num_boost_round=500, verbose_eval=50, dval=None):
    evals = [(dtrain, "train")]

    # Add validation dataset to the evaluations if provided
    if dval:
        evals.append((dval, "validation"))

    # Train the XGBoost model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        verbose_eval=verbose_eval,
        custom_metric=f1_eval
    )
    return model


# custom f1 evaluation metric
def f1_eval(y_pred, dtrain, threshold=0.5):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, y_pred > threshold)
    return 'f1_err', err


# For double checking that the columns are indeed floats
def to_float(x):
    if type(x) == str: return x
    try: return float(x)
    except: return np.nan


def prepare_data(set_number, perform_regression, objective, include_val=True, remove_curve_outliers=False):
    #df = get_df_merged(force_rebuild=False)     # use stored version if available
    df = pd.read_parquet('export_dfs/df_merged.parquet')

    if remove_curve_outliers:
        # remove some outliers from the ivol curve parametrisations
        if 'OPTION_PRC::CUSTOM_Moneyness_Impl::error_MSE' in df.columns:
            cat_cols = get_categories_columns(df, 'OPTION_PRC::CUSTOM_Moneyness_Impl::')
            for catcol in cat_cols:
                df.loc[df['OPTION_PRC::CUSTOM_Moneyness_Impl::error_MSE'] > 0.002, catcol] = np.nan
        if 'OPTION_PRC::SPLINE_Moneyness_Impl::error_MSE' in df.columns:
            cat_cols = get_categories_columns(df, 'OPTION_PRC::CUSTOM_Moneyness_Impl::')
            for catcol in cat_cols:
                df.loc[df['OPTION_PRC::SPLINE_Moneyness_Impl::error_MSE'] > 0.002, catcol] = np.nan
        if 'OPTION_PRC::SPLINE_Moneyness_Impl::error_interp_spl' in df.columns:
            cat_cols = get_categories_columns(df, 'OPTION_PRC::SPLINE_Moneyness_Impl::')
            for catcol in cat_cols:
                df.loc[df['OPTION_PRC::SPLINE_Moneyness_Impl::error_interp_spl'] > 0.001, catcol] = np.nan
        if 'OPTION_PRC::CUSTOM_Moneyness_Impl::error_interp_spl' in df.columns:
            cat_cols = get_categories_columns(df, 'OPTION_PRC::CUSTOM_Moneyness_Impl::')
            for catcol in cat_cols:
                df.loc[df['OPTION_PRC::CUSTOM_Moneyness_Impl::error_interp_spl'] > 0.001, catcol] = np.nan

    # only keep data where 'one_month_lookahead_vol' exists
    df = df.dropna(subset=['one_month_lookahead_vol'], axis='rows')
    df = df.pivot(columns='ticker', index='date')
    df = swap_levels(df)
    for ticker in df.columns.levels[0]:
        df.loc[:, (ticker, 'ATM_ahead')] = df[ticker]['OPTION_VOLSURF::GEN::atm_ivol_custom'].shift(-21)
        df.loc[:, (ticker, 'median_volatility')] = df.loc[:, (ticker, 'HIST::Volatility')].rolling(window=63).median()
    df = df.ffill()
    df = unpivot(df)
    df = df.dropna(subset=['one_month_lookahead_vol'], axis='rows')
    #df = df.fillna(0)

    feature_set = get_set(df, set_number)
    if not perform_regression:
        Hist_vol = df['HIST::Volatility']
        # feature_set = list(set(feature_set) - {'HIST::Volatility'} | {'ATM_ahead', 'median_volatility'})
        feature_set = list( set(feature_set) | {'ATM_ahead', 'median_volatility'} )
    df = df[feature_set]

    # Decide on target
    if perform_regression:
        df['target'] = df['one_month_lookahead_vol']
        df = df.drop(columns='one_month_lookahead_vol')
    else:
        # choose objective
        assert objective in {1, 2, 3, 4}
        if objective == 1:
            df['target'] = df['one_month_lookahead_vol'] > Hist_vol
        if objective == 2:
            df['target'] = df['one_month_lookahead_vol'] > df['median_volatility']
        if objective == 3:
            df['target'] = df['one_month_lookahead_vol'] > df['ATM_ahead']
        if objective == 4:
            df['target'] = df['one_month_lookahead_vol'] > 0.75*df['ATM_ahead']

        df = df.drop(columns=['one_month_lookahead_vol', 'ATM_ahead', 'median_volatility'])

    # If we want to split into TRAIN - VALIDATION - TEST
    if include_val:
        # Perform train-test split
        df_train = df[(df['date'] < '2017-01-01')]
        df_val = df[ (df['date'] >= '2017-01-01') & (df['date'] < '2018-07-01')]
        df_test = df[df['date'] >= '2018-07-01']

        X_train, y_train = get_variables(df_train)
        X_val, y_val = get_variables(df_val)
        X_test, y_test = get_variables(df_test)

        # Adjust data type
        cats = {'GICS Sector', 'month_number', 'next_event_less_30_days'}

        # Convert to Pandas category
        for col in X_train.columns:
            if col in cats:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')
                X_test[col] = X_test[col].astype('category')
            else:
                X_train[col] = X_train[col].apply(to_float)
                X_val[col] = X_val[col].apply(to_float)
                X_test[col] = X_test[col].apply(to_float)

        return (df, df_test, df_train, df_val), (X_train, y_train), (X_val, y_val), (X_test, y_test)
    # If we want to split into TRAIN - TEST
    else:
        # Perform train-test split
        df_train = df[df['date'] < '2018-01-01']
        df_test = df[df['date'] >= '2018-01-01']

        X_train, y_train = get_variables(df_train)
        X_test, y_test = get_variables(df_test)

        # Adjust data type
        cats = {'GICS Sector', 'month_number', 'next_event_less_30_days'}

        # Convert to Pandas category
        for col in X_train.columns:
            if col in cats:
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')
            else:
                X_train[col] = X_train[col].apply(to_float)
                X_test[col] = X_test[col].apply(to_float)

        return (df, df_test, df_train), (X_train, y_train), (X_test, y_test)


########################################################################################################################
# Main part
########################################################################################################################

if __name__ == '__main__':

    from imblearn.under_sampling import (RandomUnderSampler, EditedNearestNeighbours,
                                         RepeatedEditedNearestNeighbours, OneSidedSelection)
    from imblearn.combine import SMOTEENN
    from plotting import plot_confusion_matrix, plot_binary_classification_result
    from plotting import plot_regression_result

    # FLAGS
    # flag for regression or classification
    # perform_regression = False
    # classification threshold
    threshold = 0.3
    perform_regression = False
    set_number = 1


    dfs, train_data, val_data, test_data = prepare_data(set_number=set_number,
                                                        perform_regression=perform_regression,
                                                        objective=1)

    all_df, df_test, df_train, df_val = dfs
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # if we use resampler we have to have strictly numerical data --> no categorical datatypes
    my_dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
    my_dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    my_dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)

    # setup model parameters

    params = {
        "tree_method": "hist", 'booster': 'dart', # 'booster': 'gbtree', #'n_estimators': 500, # general
        'max_depth': 4, "rate_drop": 0.3, # 'subsample': 0.5 #'colsample_bytree': 1, # complexity
        # 'lambda': 4, 'alpha': 4 # L2 & L1 regularisation
        }

    if perform_regression:
        params["objective"] = "reg:squarederror"
    else:
        params["objective"] = "binary:logistic"
        # adjust for class imbalance
        ratio = float(len(y_train[y_train == 0])) / len(y_train[y_train == 1])
        print(ratio)
        params['scale_pos_weight'] = ratio

    # train xgboost model
    if perform_regression:
        model = xgboost_train_regression(dtrain=my_dtrain, params=params,
                                         num_boost_round=50, dval=my_dval, verbose_eval=1)
    else:
        model = xgboost_train_classification(dtrain=my_dtrain, params=params,
                                             num_boost_round=30, dval=my_dval, verbose_eval=1)

    # make prediction on test and train sets
    preds_test = model.predict(my_dtest)
    preds_train = model.predict(my_dtrain)
    preds_val = model.predict(my_dval)

    # calculate loss on test set
    if perform_regression:
        mse = mean_squared_error(y_test, preds_test, squared=True)
        print(f"MSE of the base model (Test set): {mse:.3f}")
    else:
        f1_test = f1_score(y_test, preds_test>threshold)
        f1_train = f1_score(y_train, preds_train>threshold)
        f1_val = f1_score(y_val, preds_val>threshold)
        print(f"F1 score of the base model (Test set): {f1_test:.3f}")
        print(f"F1 score of the base model (Train set): {f1_train:.3f}")
        print(f"F1 score of the base model (Validation set): {f1_val:.3f}")

    img_dir = 'Images/Week 3/Temp/'
    if perform_regression:
        tickers_to_plot = ['AAPL', 'MSFT', 'GS']
        for i, my_ticker in enumerate(tickers_to_plot):
            # plot model performance on training, validation and test set
            plot_regression_result(ticker=my_ticker, all_df=all_df, my_df=df_test,
                                   my_preds=preds_test, dataset_label='Test Set', save_dir=img_dir+f'{3*i + 1}.png')
            plot_regression_result(ticker=my_ticker, all_df=all_df, my_df=df_train,
                                   my_preds=preds_train, dataset_label='Train Set', save_dir=img_dir+f'{3*i + 2}.png')
            plot_regression_result(ticker=my_ticker, all_df=all_df, my_df=df_val,
                                   my_preds=preds_val, dataset_label='Validation Set', save_dir=img_dir + f'{3*i + 3}.png')

        # plot feature importance (top 10)
        plt.figure(figsize=(12, 6), dpi=500)
        plot_importance(model, importance_type='weight', max_num_features=10, ax=plt.gca())
        plt.title('Feature Importance (weight) for XGBoost Regression')
        plt.tight_layout()
        plt.savefig(img_dir + 'feature_importance_weight.png')
        plt.show()

        plt.figure(figsize=(12, 6), dpi=500)
        plot_importance(model, importance_type='gain', max_num_features=10, ax=plt.gca())
        plt.title('Feature Importance (gain) for XGBoost Regression')
        plt.tight_layout()
        plt.savefig(img_dir + 'feature_importance_gain.png')
        plt.show()

        plt.figure(figsize=(12, 6), dpi=500)
        plot_importance(model, importance_type='cover', max_num_features=10, ax=plt.gca())
        plt.title('Feature Importance (cover) for XGBoost Regression')
        plt.tight_layout()
        plt.savefig(img_dir + 'feature_importance_cover.png')
        plt.show()

    # classification case
    else:
        tickers_to_plot = ['AAPL', 'MSFT', 'GS']
        for i, my_ticker in enumerate(tickers_to_plot):
            plot_binary_classification_result(ticker=my_ticker, all_df=all_df, my_df=df_test, my_preds=preds_test>threshold,
                                              dataset_label='Test Set', save_dir=img_dir + f'{3 * i + 1}.png')
            plot_binary_classification_result(ticker=my_ticker, all_df=all_df, my_df=df_train, my_preds=preds_train>threshold,
                                              dataset_label='Train Set', save_dir=img_dir + f'{3 * i + 2}.png')
            plot_binary_classification_result(ticker=my_ticker, all_df=all_df, my_df=df_val, my_preds=preds_val>threshold,
                                              dataset_label='Validation Set', save_dir=img_dir + f'{3 * i + 3}.png')

        # Plot confusion matrix test set
        plot_confusion_matrix(y_test, preds_test > threshold,
                              title=f'Confusion Matrix -- Test Set -- F1 Score: {f1_test:.3f}',
                              save_dir=img_dir + '1.png')

        # Plot confusion matrix train set
        plot_confusion_matrix(y_train, preds_train > threshold,
                              title=f'Confusion Matrix -- Train Set -- F1 Score: {f1_train:.3f}',
                              save_dir=img_dir + '2.png')

        # Plot confusion matrix train set
        plot_confusion_matrix(y_val, preds_val > threshold,
                              title=f'Confusion Matrix -- Validation Set -- F1 Score: {f1_val:.3f}',
                              save_dir=img_dir + '3.png')

    # print the most important features
    feature_important = model.get_score(importance_type='gain')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    features_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    features_list = list(features_df[features_df['score']>25].index)
    features_list = [f'{feature}' for feature in features_list]
    print(features_list)
