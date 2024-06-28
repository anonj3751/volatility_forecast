# Code used for feature extraction and volatility forecasting

The files included are:

economic_data.py:
- merge various datasets of financial/economic data

feature_sets.py:
- define different feature sets
- includes functions to handle different feature categories in an organised way

GridSearch.py:
- perform gridsearch for XGBoost and using Cross-Validation

helper.py:
- auxilary functions

JSON extract.py:
- extract earning release dates from SEC EDGAR database of JSON files

merge_databases.py:
- merge different databases + compute historic volatility indicator

natural_cubic_splines.py:
- helper file for computing natural cubic splines

Option_Features.py:
- Combine databases of features collected for option prices and implied volatility surfaces

Option_Price_Features.py:
- Extract several features from option price data from OptionMetrics

Option_Volsurf_Features.py:
- Extract several features from implied volatility surface data from OptionMetrics

plotting.py:
- Helper file for plotting

stock_data.py:
- Import stock data and financial ratios

xgboost_fit.py:
- Fit regression or classification model with XGBoost

earning_data_factset.py:
- collect data manually downloaded from FactSet and merge to single database
