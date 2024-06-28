import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stock_data import get_stock_data
import matplotlib as mpl
from stock_data import read_tickers
from earning_data_factset import get_earning_data
from Option_Features import get_option_data
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from helper import unpivot
from statsmodels.tsa.ar_model import AutoReg

from helper import compute_adfuller
from helper import get_files_in_folder
from arch import arch_model
from economic_data import get_economic_data
from plotting import plot_ACF
from plotting import plot_PACF

debug_plotting = False

########################################################################################################################
# Helper functions
########################################################################################################################

# Calculates difference in trading days to next company event
def calculate_days_until_event(df, ticker):
    if 'event_happened' not in df[ticker].columns:
        return df
    # Mark dates where events happened
    df.loc[:, (ticker, 'event_date')] = df[ticker].index.get_level_values('date').where(df[ticker]['event_happened'] == 1)
    # Forward fill the 'event_date' within each group
    df.loc[:, (ticker, 'event_date')] = df[ticker]['event_date'].bfill()
    # Calculate days until next event
    df.loc[:, (ticker, 'days_to_event')] = df[ticker]['event_date'] - df[ticker].index.get_level_values('date')
    df.loc[:, (ticker, 'days_to_event')] = df[ticker]['days_to_event'].apply(lambda my_date: my_date.days)
    # Convert to estimated number of trading days
    df.loc[:, (ticker, 'days_to_event')] *= (250/365)
    df.loc[:, (ticker, 'days_to_event')] = df.loc[:, (ticker, 'days_to_event')].round()
    # Calculate dummy variable whether next event is in next 21 trading days
    df.loc[:, (ticker, 'next_event_less_21_days')] = (df[ticker]['days_to_event'] < 21).astype(int)
    df = df.drop( columns=[(ticker, 'event_date'), (ticker, 'event_happened')] )
    return df


# Create column for volatility in 30 trading days (value to predict)
def find_closest_date(date_index, current_date, days):
    target_date = current_date + pd.DateOffset(days=days)
    if days >= 0: mask = date_index >= target_date
    else: mask = date_index <= target_date

    if mask.any(): return date_index[mask][0]
    return None


# perform ACF and PACF analysis (plotting) and ADFuller test for stationarity
def acf_pacf_stationarity_analysis_volatility(df, tickers, num_of_lags=30):
    for ticker in tickers:
        my_data = df[ticker]['Volatility'].dropna()  # previously checked that data does not have gaps

        # Save ACF and PACF plots to disk (skip if the images already exist on disk)
        acf_files = get_files_in_folder('Images/ACF_plots', 'png')
        pacf_files = get_files_in_folder('Images/PACF_plots', 'png')
        if len(acf_files)==0:
            plot_ACF(my_data, num_of_lags, ticker, show=False, basepath='Images/ACF_plots')
        if len(pacf_files)==0:
            plot_PACF(my_data, num_of_lags, ticker, show=False, basepath='Images/PACF_plots')

        # Perform ADFuller
        p_val = compute_adfuller(my_data, print_details=False)
        if p_val > 0.05:
            print(f'Cannot reject existence of unit root for: {ticker} --> p_val = {p_val} --> might be stationary')


# Function to fit the AR model and make predictions
def fit_AR_model(data, lags, max_forecast):
    ar_data = data.copy()
    ar_data = ar_data.rename('volatility')
    model = AutoReg(ar_data, lags=lags).fit()
    target_day = len(ar_data) - 1 + max_forecast
    predictions = model.predict(start=len(ar_data), end=target_day)
    # predictions[ predictions < 0 ] = 0   # volatility cannot be negative!
    return np.abs(predictions)


# Function to fit the HAR model and make predictions
def fit_HAR_model(data, max_forecast):
    har_data = data.copy()
    har_data = pd.DataFrame(har_data.rename('tseries'))
    # Lagged features for HAR model (daily, weekly, and monthly)
    har_data['tseries_lag_1'] = har_data.tseries
    har_data['tseries_lag_5'] = har_data.tseries.rolling(window=5).mean()
    har_data['tseries_lag_21'] = har_data.tseries.rolling(window=21).mean()
    # Prepare the data for forecasting
    last_row = har_data.iloc[-1]
    forecast_data = np.array([
        last_row['tseries_lag_1'],
        last_row['tseries_lag_5'],
        last_row['tseries_lag_21']
    ]).reshape(1, -1)
    # Lag the 1, 5 and 21 day values
    har_data['tseries_lag_1'] = har_data['tseries_lag_1'].shift(1)
    har_data['tseries_lag_5'] = har_data['tseries_lag_5'].shift(1)
    har_data['tseries_lag_21'] = har_data['tseries_lag_21'].shift(1)
    har_data = har_data.dropna()  # Drop the rows with NaN values due to lagging
    # Fit the HAR model
    X = har_data[['tseries_lag_1', 'tseries_lag_5', 'tseries_lag_21']]
    y = har_data['tseries']
    model = AutoReg(y, lags=0, exog=X).fit()
    # Forecast future values
    predictions = np.zeros(max_forecast)  # will be populated in the for-loop
    for i in range(max_forecast):
        next_pred = model.predict(start=len(y), end=len(y), exog_oos=forecast_data).iloc[0]
        predictions[i] = next_pred
        # Update forecast_data for the next prediction
        new_lag_1 = next_pred
        new_lag_5 = (forecast_data[0, 1] * 4 + next_pred) / 5  # Rolling mean update
        new_lag_21 = (forecast_data[0, 2] * 20 + next_pred) / 21  # Rolling mean update
        forecast_data = np.array([new_lag_1, new_lag_5, new_lag_21]).reshape(1, -1)

    return np.abs(predictions)  # volatility cannot be negative


# Function to fit a GARCH or EGARCH model and make a prediction for future estimated volatility
def fit_GARCH_model(data, p, q, type, dist, max_forecast):
    assert type in {'Garch', 'EGARCH'}
    ret_data = data.copy()
    ret_data = ret_data.rename('returns')
    model = arch_model(ret_data, mean='Zero', vol=type, p=p, q=q, dist=dist).fit(disp='off')
    if type == 'EGARCH':
        cond_var_predictions = model.forecast(horizon=max_forecast, method="simulation").variance
    else:
        cond_var_predictions = model.forecast(horizon=max_forecast).variance

    annualized_var = np.sum(np.array(cond_var_predictions)) * (250 / max_forecast)
    return model.conditional_volatility, np.sqrt(annualized_var)


def unpack_array_df(df, ticker, col_to_unpack, new_columns):
    # Handle NaN case --> replace rows where NaN with arrays of NaNs of the appropriate length
    df[(ticker, col_to_unpack)] = df[(ticker, col_to_unpack)].apply(
        lambda row: row if type(row) == np.ndarray else np.full(8, np.nan)
    )
    # unpack the column
    array_col = df.loc[:, (ticker, col_to_unpack)]
    unpacked_df = pd.DataFrame(array_col.tolist(), index=df.index, columns=new_columns)  # unpack arrays
    for col in unpacked_df.columns:
        df[(ticker, col)] = unpacked_df[col]  # create new columns
    df = df.drop(columns=[(ticker, col_to_unpack)])  # drop old column
    return df


def calculate_df_merged():
    # suppress warnings (need for GARCH fitting)
    #warnings.filterwarnings('ignore', category=UserWarning)

    ####################################################################################################################
    # Part 1
    ####################################################################################################################

    t0 = time.time()

    # load stocks data
    stock_df = get_stock_data()

    # load earning dates data
    earnings_df = get_earning_data()

    # load extracted options features
    options_price_df, options_volsurf_df = get_option_data()

    # get columns to exclude for earnings_df, options_price_df and options_volsurf_df
    tickers_exclude = read_tickers("TICKERS/TICKER_exclude.txt")        # tickers to exclude
    earnings_df_exclude = [col for col in earnings_df.columns if col[0] in tickers_exclude]
    options_price_df_exclude = [col for col in options_price_df.columns if col[0] in tickers_exclude]
    options_volsurf_df_exclude = [col for col in options_volsurf_df.columns if col[0] in tickers_exclude]

    # exclude tickers with too little data
    earnings_df = earnings_df.drop(columns=earnings_df_exclude)
    options_volsurf_df = options_volsurf_df.drop(columns=options_volsurf_df_exclude, errors='ignore')
    options_price_df = options_price_df.drop(columns=options_price_df_exclude, errors='ignore')

    # remove unused levels
    earnings_df.columns = earnings_df.columns.remove_unused_levels()
    options_volsurf_df.columns = options_volsurf_df.columns.remove_unused_levels()
    options_price_df.columns = options_price_df.columns.remove_unused_levels()

    # left join stock_df, (options_df, earnings_df)
    df = stock_df.join([options_price_df, options_volsurf_df, earnings_df])
    df = df.sort_index(axis=1, level=0)

    # Calculate volatility
    # .shift(1) is to prevent lookahead bias [we don't have the current return when calculating today's volatility]
    # So use closing prices from until yesterday to estimate volatility
    window_size = 21
    rolling_volatility = df.xs('LogRet', level=1, axis=1).rolling(window=window_size).std().shift(1) * np.sqrt(250)
    for ticker in rolling_volatility.columns:
        df[(ticker, 'Volatility')] = rolling_volatility[ticker]
    df = df.sort_index(axis=1)


    ####################################################################################################################
    # Part 1.5 --> perform statistical analysis for the data
    ####################################################################################################################
    # Plot autocorrelation and partial autocorrelation tests
    acf_pacf_stationarity_analysis_volatility(df, tickers=df.columns.levels[0], num_of_lags=60)


    ####################################################################################################################
    # Part 2 -- Compute Volatility Features
    ####################################################################################################################
    max_allowed_volatility = 2.  # we need to cap some volatility estimations (as some tend to explode)
    for ticker in df.columns.levels[0]:
        t00 = time.time()

        ################################################################################################################
        # Target: One month lookahead
        ################################################################################################################
        df.loc[:, (ticker, 'one_month_lookahead_vol')] = df.loc[:, (ticker, 'Volatility')].shift(-21)


        ################################################################################################################
        # Add EVENT information in two features:
        # 1) 'days_until_event' --> number of trading days until next event (estimated trading days)
        # 2) 'next_event_less_21_days' --> indicator whether in the next 21 trading days an event will happen
        ################################################################################################################
        df = calculate_days_until_event(df, ticker)


        ################################################################################################################
        # Past Volatility Estimators (AutoRegressive and ARCH-type)
        ################################################################################################################
        # Compute the historical volatility (all historical data available at that point in time)
        df.loc[:, (ticker, 'HIST::HA_Volatility')] = df.loc[:, (ticker, 'Volatility')].expanding().mean()
        # Compute the moving average volatility (21 days)
        df.loc[:, (ticker, 'HIST::MA_Volatility')] = df.loc[:, (ticker, 'Volatility')].rolling(window=21).mean()
        # Compute the exponentially weighted moving average
        beta = 0.95
        df.loc[:, (ticker, 'HIST::EWMA_Volatility')] = df.loc[:, (ticker, 'Volatility')].ewm(alpha=1 - beta,
                                                                                       adjust=True).mean()
        first_valid_volatility_idx = df.loc[:, (ticker, 'Volatility')].first_valid_index()
        first_valid_volatility_idx = df.index.get_loc(first_valid_volatility_idx)   # convert to iloc idx
        max_forecast = 21
        for idx in range(4+first_valid_volatility_idx, len(df)):
            data_volatility = df.loc[:, (ticker, 'Volatility')].iloc[first_valid_volatility_idx:idx]
            if data_volatility.isna().any(): # cannot fit AR model with nan values
                continue
            # plotting
            if debug_plotting and idx>140:
                mpl.style.use('seaborn-v0_8')
                plt.figure(figsize=(12, 6), dpi=200)
                plt.plot(range(len(data_volatility)), data_volatility)

            # Fit AR model for estimated volatility and make forecasts
            N = min( 30, int((idx - first_valid_volatility_idx - 2) / 2) )   # N := maximal possible lag but not more than 30 lags
            ARN_volatility_preds = fit_AR_model(data_volatility, N, max_forecast)
            ARN_volatility_1, ARN_volatility_21 = ARN_volatility_preds.iloc[ [0, -1] ]
            df.loc[df.index[idx], (ticker, 'HIST::AR(N)_volatility_1')] = min(ARN_volatility_1, max_allowed_volatility)
            df.loc[df.index[idx], (ticker, 'HIST::AR(N)_volatility_21')] = min(ARN_volatility_21, max_allowed_volatility)
            # Fit second AR model ( AR(1) ) for estimated volatility and make forecasts
            AR1_volatility_preds = fit_AR_model(data_volatility, 1, max_forecast)
            AR1_volatility_1, AR1_volatility_21 = AR1_volatility_preds.iloc[ [0, -1] ]
            df.loc[df.index[idx], (ticker, 'HIST::AR(1)_volatility_1')] = min(AR1_volatility_1, max_allowed_volatility)
            df.loc[df.index[idx], (ticker, 'HIST::AR(1)_volatility_21')] = min(AR1_volatility_21, max_allowed_volatility)
            # Plotting
            if debug_plotting and idx>140:
                plt.plot(range(len(data_volatility)-1, len(data_volatility) + len(ARN_volatility_preds)),
                         [*[data_volatility.iloc[-1]], *list(ARN_volatility_preds)],
                         label='ARN_volatility_preds')
                plt.plot(range(len(data_volatility) - 1, len(data_volatility) + len(AR1_volatility_preds)),
                         [*[data_volatility.iloc[-1]], *list(AR1_volatility_preds)],
                         label='AR1_volatility_preds')

            # Fit HAR model for estimated volatility and make forecasts
            if len(data_volatility) > 21 + 4:      # need at least 5 observations (which aren't NaN)
                HAR_volatility_preds = fit_HAR_model(data_volatility, max_forecast)
                HAR_volatility_1 = HAR_volatility_preds[0]
                HAR_volatility_21 = HAR_volatility_preds[-1]
                df.loc[df.index[idx], (ticker, 'HIST::HAR_volatility_1')] = min(HAR_volatility_1, max_allowed_volatility)
                df.loc[df.index[idx], (ticker, 'HIST::HAR_volatility_21')] = min(HAR_volatility_21, max_allowed_volatility)
                # plotting
                if debug_plotting and idx>140:
                    plt.plot(range(len(data_volatility)-1, len(data_volatility) + len(HAR_volatility_preds)),
                             [*[data_volatility.iloc[-1]] , *list(HAR_volatility_preds)],
                             label='HAR_volatility_preds')

            # Fit GARCH and EGARCH models for conditional variance
            first_valid_ret_idx = df.loc[:, (ticker, 'LogRet')].first_valid_index()
            first_valid_ret_idx = df.index.get_loc(first_valid_ret_idx)  # convert to iloc idx
            data_ret = df.loc[:, (ticker, 'LogRet')].iloc[first_valid_ret_idx:idx]
            if len(data_ret) > 21 + 4:     # need at least 5 observations (which aren't NaN)
                cond_vols_GARCH, annualized_pred_GARCH = fit_GARCH_model(data_ret, 1, 1,
                                                                             'Garch', 'Normal',
                                                                             max_forecast)
                cond_vols_EGARCH, annualized_pred_EGARCH = fit_GARCH_model(data_ret, 1, 1,
                                                                               'EGARCH', 'Normal',
                                                                               max_forecast)
                df.loc[df.index[idx], (ticker, 'HIST::GARCH_pred')] = annualized_pred_GARCH
                df.loc[df.index[idx], (ticker, 'HIST::GARCH_crt_vol')] = cond_vols_GARCH[-1]*np.sqrt(250)     # annualised
                df.loc[df.index[idx], (ticker, 'HIST::EGARCH_pred')] = annualized_pred_EGARCH
                df.loc[df.index[idx], (ticker, 'HIST::EGARCH_crt_vol')] = cond_vols_EGARCH[-1]*np.sqrt(250)     # annualised
                # The arrays 'cond_vols_GARCH' and 'cond_vols_EGARCH' are also used for the HAR models

            # Fit HAR models for realized variance
            if len(data_ret) > 21 + 4:      # need at least 5 observations (which aren't NaN)
                # DAILY SQUARED LOG-RETURNS
                HAR_ret2_real_variance_preds = fit_HAR_model(np.abs(data_ret), max_forecast) ** 2
                HAR_ret2_real_variance_21 = np.sum(HAR_ret2_real_variance_preds) * (250/21)     # annualised
                df.loc[df.index[idx], (ticker, 'HIST::HAR_ret2_real_variance_21')] = min(np.sqrt(HAR_ret2_real_variance_21),
                                                                         max_allowed_volatility)
                # CONDITIONAL VARIANCE FROM GARCH
                HAR_GARCH_real_variance_preds = fit_HAR_model(cond_vols_GARCH, max_forecast) ** 2
                HAR_GARCH_real_variance_21 = np.sum(HAR_GARCH_real_variance_preds) * (250/21)    # annualised
                df.loc[df.index[idx], (ticker, 'HIST::HAR_GARCH_real_variance_21')] = min(np.sqrt(HAR_GARCH_real_variance_21),
                                                                          max_allowed_volatility)
                # CONDITIONAL VARIANCE FROM EGARCH
                HAR_EGARCH_real_variance_preds = fit_HAR_model(cond_vols_EGARCH, max_forecast) ** 2
                HAR_EGARCH_real_variance_21 = np.sum(HAR_EGARCH_real_variance_preds) * (250/21)  # annualised
                df.loc[df.index[idx], (ticker, 'HIST::HAR_EGARCH_real_variance_21')] = min(np.sqrt(HAR_EGARCH_real_variance_21),
                                                                           max_allowed_volatility)

            if debug_plotting and idx>140:
                plt.title(f'Volatility predictions: {ticker}')
                plt.legend()
                plt.show()


        ################################################################################################################
        # Unpack the curve parameterization
        ################################################################################################################
        ## CUSTOM PARAMETRISATION
        # unpack parameters of custom smile interpolation -- moneyness <-> ivol
        arr_col_name = 'smile_0::param_custom_moneyness_impl'
        new_columns = ['x_min', 'x_max', 'a', 'b', 'c', 'd', 'centre', 'centre_y', 'alpha']
        new_columns = ['OPTION_PRC::CUSTOM_Moneyness_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option price data
        arr_col_name = 'smile_volsurf::param_custom_moneyness_impl'
        new_columns = ['x_min', 'x_max', 'a', 'b', 'c', 'd', 'centre', 'centre_y', 'alpha']
        new_columns = ['OPTION_VOLSURF::CUSTOM_Moneyness_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option volsurf data

        # unpack parameters of custom smile interpolation --- nlmoneyness <-> ivol
        arr_col_name = 'smile_0::param_custom_nlmoneyness_impl'
        new_columns = ['x_min', 'x_max', 'a', 'b', 'c', 'd', 'centre', 'centre_y', 'alpha']
        new_columns = ['OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option price data
        arr_col_name = 'smile_volsurf::param_custom_nlmoneyness_impl'
        new_columns = ['x_min', 'x_max', 'a', 'b', 'c', 'd', 'centre', 'centre_y', 'alpha']
        new_columns = ['OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option volsurf data

        ## SPLINE PARAMETRISATION
        # unpack parameters of spline smile interpolation --- moneyness <-> ivol
        arr_col_name = 'smile_0::param_spline_moneyness_impl'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_PRC::SPLINE_Moneyness_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option price data
        arr_col_name = 'smile_volsurf::param_spline_moneyness_impl'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_VOLSURF::SPLINE_Moneyness_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option volsurf data

        # unpack parameters of spline smile interpolation --- moneyness <-> price
        arr_col_name = 'smile_0::param_spline_moneyness_price'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_PRC::SPLINE_Moneyness_Prc::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option price data
        arr_col_name = 'smile_volsurf::param_spline_moneyness_price'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_VOLSURF::SPLINE_Moneyness_Prc::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option volsurf data

        # unpack parameters of spline smile interpolation --- nlmoneyness <-> ivol
        arr_col_name = 'smile_0::param_spline_nlmoneyness_impl'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option price data
        arr_col_name = 'smile_volsurf::param_spline_nlmoneyness_impl'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option volsurf data

        # unpack parameters of spline smile interpolation --- nlmoneyness <-> price
        arr_col_name = 'smile_0::param_spline_nlmoneyness_price'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option price data
        arr_col_name = 'smile_volsurf::param_spline_nlmoneyness_price'
        new_columns = ['x_min', 'x_max', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        new_columns = ['OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::' + col for col in new_columns]
        df = unpack_array_df(df, ticker, arr_col_name, new_columns)     # option volsurf data

        print(f"Time for {ticker}:  {time.time() - t00}")

    ################################################################################################################
    # Part 3 -- Rename columns to organize in categories
    ################################################################################################################
    # First some manual renaming
    # rename dict --> specifies what & how to rename
    rename_dict = {
        ## CRSP variables (derived from CRSP database)
        'ASK-BID': 'CRSP::ASK-BID',
        'ASKHI-BIDLO': 'CRSP::ASKHI-BIDLO',
        'LogRet': 'CRSP::LogRet',
        'SHROUT': 'CRSP::SHROUT',
        'VOL': 'CRSP::Stock_Volume',
        'ewretd': 'CRSP::ewretd',
        'ewretx': 'CRSP::ewretx',
        'vwretd': 'CRSP::vwretd',
        'vwretx': 'CRSP::vwretx',

        ## Variables from Option Price File
        # 'General' variables
        'smile_0::atm_ivol_custom': 'OPTION_PRC::GEN::atm_ivol_custom',
        'smile_0::atm_ivol_spline': 'OPTION_PRC::GEN::atm_ivol_spline',
        'smile_0::atm_slope_custom': 'OPTION_PRC::GEN::atm_slope_custom',
        'smile_0::atm_slope_spline': 'OPTION_PRC::GEN::atm_slope_spline',
        'smile_0::ba_spread_avg': 'OPTION_PRC::GEN::ba_spread_avg',
        'smile_0::ba_spread_avg_call': 'OPTION_PRC::GEN::ba_spread_avg_call',
        'smile_0::ba_spread_avg_put': 'OPTION_PRC::GEN::ba_spread_avg_put',
        'smile_0::ba_spread_median': 'OPTION_PRC::GEN::ba_spread_median',
        'smile_0::ba_spread_median_call': 'OPTION_PRC::GEN::ba_spread_median_call',
        'smile_0::ba_spread_median_put': 'OPTION_PRC::GEN::ba_spread_median_put',
        'smile_0::ivol_spread': 'OPTION_PRC::GEN::ivol_spread',
        'smile_0::number_contracts_call': 'OPTION_PRC::GEN::number_contracts_call',
        'smile_0::number_contracts_put': 'OPTION_PRC::GEN::number_contracts_put',
        'smile_0::time_to_mat': 'OPTION_PRC::GEN::time_to_mat',
        'smile_0::total_volume_call': 'OPTION_PRC::GEN::total_volume_call',
        'smile_0::total_volume_put': 'OPTION_PRC::GEN::total_volume_put',
        # 'Model-free' variables
        'smile_0::mf_ivol_cboe': 'OPTION_PRC::MF::ivol_cboe',
        'smile_0::mf_ivol_raw_kurt': 'OPTION_PRC::MF::kurtosis',
        'smile_0::mf_ivol_raw_kurt_trapz': 'OPTION_PRC::MF::kurtosis_trapz',
        'smile_0::mf_ivol_raw_skew': 'OPTION_PRC::MF::skew',
        'smile_0::mf_ivol_raw_skew_trapz': 'OPTION_PRC::MF::skew_trapz',
        'smile_0::mf_ivol_raw_vol': 'OPTION_PRC::MF::volatility',
        'smile_0::mf_ivol_raw_vol_trapz': 'OPTION_PRC::MF::volatility_trapz',
        # 'Error' variables
        'smile_0::error_MSE_model_custom_moneyness_impl': 'OPTION_PRC::CUSTOM_Moneyness_Impl::error_MSE',
        'smile_0::error_MSE_model_custom_nlmoneyness_impl': 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::error_MSE',
        'smile_0::error_MSE_model_spline_moneyness_impl': 'OPTION_PRC::SPLINE_Moneyness_Impl::error_MSE',
        'smile_0::error_MSE_model_spline_moneyness_price': 'OPTION_PRC::SPLINE_Moneyness_Prc::error_MSE',
        'smile_0::error_MSE_model_spline_nlmoneyness_impl': 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::error_MSE',
        'smile_0::error_MSE_model_spline_nlmoneyness_price': 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::error_MSE',
        'smile_0::error_interp_lin_model_custom_moneyness_impl': 'OPTION_PRC::CUSTOM_Moneyness_Impl::error_interp_lin',
        'smile_0::error_interp_lin_model_custom_nlmoneyness_impl': 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::error_interp_lin',
        'smile_0::error_interp_lin_model_spline_moneyness_impl': 'OPTION_PRC::SPLINE_Moneyness_Impl::error_interp_lin',
        'smile_0::error_interp_lin_model_spline_moneyness_price': 'OPTION_PRC::SPLINE_Moneyness_Prc::error_interp_lin',
        'smile_0::error_interp_lin_model_spline_nlmoneyness_impl': 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::error_interp_lin',
        'smile_0::error_interp_lin_model_spline_nlmoneyness_price': 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::error_interp_lin',
        'smile_0::error_interp_spl_model_custom_moneyness_impl': 'OPTION_PRC::CUSTOM_Moneyness_Impl::error_interp_spl',
        'smile_0::error_interp_spl_model_custom_nlmoneyness_impl': 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::error_interp_spl',
        'smile_0::error_interp_spl_model_spline_moneyness_impl': 'OPTION_PRC::SPLINE_Moneyness_Impl::error_interp_spl',
        'smile_0::error_interp_spl_model_spline_moneyness_price': 'OPTION_PRC::SPLINE_Moneyness_Prc::error_interp_spl',
        'smile_0::error_interp_spl_model_spline_nlmoneyness_impl': 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::error_interp_spl',
        'smile_0::error_interp_spl_model_spline_nlmoneyness_price': 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::error_interp_spl',

        ## Variables from Option Volatility Surface File
        # 'Model-free' variables
        'smile_volsurf::mf_ivol_raw_kurt': 'OPTION_VOLSURF::MF::kurtosis',
        'smile_volsurf::mf_ivol_raw_kurt_trapz': 'OPTION_VOLSURF::MF::kurtosis_trapz',
        'smile_volsurf::mf_ivol_raw_skew': 'OPTION_VOLSURF::MF::skew',
        'smile_volsurf::mf_ivol_raw_skew_trapz': 'OPTION_VOLSURF::MF::skew_trapz',
        'smile_volsurf::mf_ivol_raw_vol': 'OPTION_VOLSURF::MF::volatility',
        'smile_volsurf::mf_ivol_raw_vol_trapz': 'OPTION_VOLSURF::MF::volatility_trapz',
        # 'General' variables
        'smile_volsurf::ivol_spread': 'OPTION_VOLSURF::GEN::ivol_spread',
        'smile_volsurf::atm_ivol_custom': 'OPTION_VOLSURF::GEN::atm_ivol_custom',
        'smile_volsurf::atm_ivol_spline': 'OPTION_VOLSURF::GEN::atm_ivol_spline',
        'smile_volsurf::atm_slope_custom': 'OPTION_VOLSURF::GEN::atm_slope_custom',
        'smile_volsurf::atm_slope_spline': 'OPTION_VOLSURF::GEN::atm_slope_spline',
        'smile_volsurf::number_contracts_call': 'OPTION_VOLSURF::GEN::number_contracts_call',
        'smile_volsurf::number_contracts_put': 'OPTION_VOLSURF::GEN::number_contracts_put',
        # 'Error' variables
        'smile_volsurf::error_MSE_model_custom_moneyness_impl': 'OPTION_VOLSURF::CUSTOM_Moneyness_Impl::error_MSE',
        'smile_volsurf::error_MSE_model_custom_nlmoneyness_impl': 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::error_MSE',
        'smile_volsurf::error_MSE_model_spline_moneyness_impl': 'OPTION_VOLSURF::SPLINE_Moneyness_Impl::error_MSE',
        'smile_volsurf::error_MSE_model_spline_moneyness_price': 'OPTION_VOLSURF::SPLINE_Moneyness_Prc::error_MSE',
        'smile_volsurf::error_MSE_model_spline_nlmoneyness_impl': 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::error_MSE',
        'smile_volsurf::error_MSE_model_spline_nlmoneyness_price': 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::error_MSE',
        'smile_volsurf::error_interp_lin_model_custom_moneyness_impl': 'OPTION_VOLSURF::CUSTOM_Moneyness_Impl::error_interp_lin',
        'smile_volsurf::error_interp_lin_model_custom_nlmoneyness_impl': 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::error_interp_lin',
        'smile_volsurf::error_interp_lin_model_spline_moneyness_impl': 'OPTION_VOLSURF::SPLINE_Moneyness_Impl::error_interp_lin',
        'smile_volsurf::error_interp_lin_model_spline_moneyness_price': 'OPTION_VOLSURF::SPLINE_Moneyness_Prc::error_interp_lin',
        'smile_volsurf::error_interp_lin_model_spline_nlmoneyness_impl': 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::error_interp_lin',
        'smile_volsurf::error_interp_lin_model_spline_nlmoneyness_price': 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::error_interp_lin',
        'smile_volsurf::error_interp_spl_model_custom_moneyness_impl': 'OPTION_VOLSURF::CUSTOM_Moneyness_Impl::error_interp_spl',
        'smile_volsurf::error_interp_spl_model_custom_nlmoneyness_impl': 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::error_interp_spl',
        'smile_volsurf::error_interp_spl_model_spline_moneyness_impl': 'OPTION_VOLSURF::SPLINE_Moneyness_Impl::error_interp_spl',
        'smile_volsurf::error_interp_spl_model_spline_moneyness_price': 'OPTION_VOLSURF::SPLINE_Moneyness_Prc::error_interp_spl',
        'smile_volsurf::error_interp_spl_model_spline_nlmoneyness_impl': 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::error_interp_spl',
        'smile_volsurf::error_interp_spl_model_spline_nlmoneyness_price': 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::error_interp_spl',

        ## Target variable --> stays same
        'one_month_lookahead_vol': 'one_month_lookahead_vol',       # target

        ## Event variables
        'days_to_event': 'EVENT::days_to_event',
        'next_event_less_21_days': 'EVENT::next_event_less_21_days',

        ## Historical Volatility variables
        'Volatility': 'HIST::Volatility',
        'HA_Volatility': 'HIST::HA_Volatility',
        'MA_Volatility': 'HIST::MA_Volatility',
        'EWMA_Volatility': 'HIST::EWMA_Volatility',
    }
    df = df.rename(columns=rename_dict, level=1)

    # Now renaming for _Mean and _Median
    columns = list(df.columns.levels[1])
    mean_cols = [col for col in columns if '_Mean' in col ]     # filter for columns with '_Mean' in the name
    rename_dict_mean = { col: f'FinRatio_Mean::{col[:-5]}' for col in mean_cols}
    df = df.rename(columns=rename_dict_mean, level=1)
    median_cols = [col for col in columns if '_Median' in col ]     # filter for columns with '_Median' in the name
    rename_dict_median = {col: f'FinRatio_Median::{col[:-7]}' for col in median_cols}
    df = df.rename(columns=rename_dict_median, level=1)

    ####################################################################################################################
    # Part 4 -- Add ECONOMIC/FINANCIAL variables
    ####################################################################################################################
    # For merging, it's best to unpivot the dataframe first (only one level)
    df = unpivot(df)
    # Load dataframe with economic (macro) features and merge it with 'df'
    df_economic = get_economic_data()
    df = pd.merge(df, df_economic, how='left', left_on='date', right_on='Date')

    ####################################################################################################################
    # Part 5 -- Remove outliers in HIST:: group (ensure volatility values lie between [0,max_allowed_volatility])
    ####################################################################################################################
    for col in df.columns:
        if 'HIST::' in col:
            df[col] = df[col].apply( lambda x: max(0., min(max_allowed_volatility, x)) )

    ####################################################################################################################
    # Part 6 -- Save the dataframe to disk and return
    ####################################################################################################################
    df = df.drop(columns=['year-month', 'Date'])  # drop year-month column
    # sort columns alphabetically but put 'date', 'ticker' and 'one_month_lookahead_vol' first
    cols_first = ['date', 'ticker', 'one_month_lookahead_vol']
    cols_rest = sorted([col for col in df.columns if not col in cols_first])
    df = df[ cols_first + cols_rest ]
    # save to disk
    df.to_parquet('export_dfs/df_merged.parquet')
    print(f"Done with merging. Total time: {time.time() - t0}")
    return df


def get_df_merged(force_rebuild=False):
    parquet_files = get_files_in_folder('export_dfs/', 'parquet')
    if 'export_dfs/df_merged.parquet' in parquet_files and not force_rebuild:
        return pd.read_parquet('export_dfs/df_merged.parquet')

    # else we need to build the merged dataframe
    print('No copy found on disk ==> Need to recompute (this might take a while)')
    return calculate_df_merged()


########################################################################################################################
# Execution part
########################################################################################################################

if __name__=='__main__':
    df_merged = get_df_merged(force_rebuild=True)
