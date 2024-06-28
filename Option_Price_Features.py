import os
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from natural_cubic_splines import get_natural_cubic_spline_model
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from helper import custom_function
from helper import swap_levels
import argparse
from helper import get_files_in_folder

# Functions
########################################################################################################################


def clean_option_data_Prices(df, spot):
    feature_dict = {}

    # add price column
    df['price'] = (df['best_bid'] + df['best_offer']) / 2  # use mid-price

    # extract deviation from put-call parity
    strikes = set(df.strike_price)
    ivol_spreads = []
    for strike in strikes:
        calls = df[(df['strike_price'] == strike) & (df['cp_flag'] == 'C')]
        puts = df[(df['strike_price'] == strike) & (df['cp_flag'] == 'P')]
        if len(calls) > 0 and len(puts) > 0:
            ivol_spreads.append(puts.impl_volatility.iloc[0] - calls.impl_volatility.iloc[0])
    if ivol_spreads:
        feature_dict['ivol_spread'] = sum(ivol_spreads) / len(ivol_spreads)
    else:
        feature_dict['ivol_spread'] = 0.    # if there is no data on put-call spread, we set it to 0

    # only keep OTM puts and OTM calls
    OTM_puts = (df['cp_flag'] == 'P') & (df['strike_price'] * 1.01 < spot)
    OTM_calls = (df['cp_flag'] == 'C') & (df['strike_price'] * 0.99 > spot)
    df = df[OTM_puts | OTM_calls]

    # only keep contracts with volume greater than 0
    df = df[ df['volume'] > 0 ]

    return df, feature_dict


# By CBOE (and Goldman Sachs)
def modelfree_ivol_cboe(smile, r):
    df = smile.copy()
    df = df.sort_values('strike_price')
    F = df.iloc[0].ForwardPrice
    smaller_than_F = df[ df['strike_price'] <= F ]
    if len(smaller_than_F) == 0:
        return np.nan   # cannot calculate volatility
    K0 = smaller_than_F['strike_price'].iloc[-1]  # can do iloc[-1] because filtered dataframe not empty

    # setup filter for 'OTM' puts
    is_otm_put = (df['strike_price'] <= K0) & (df['cp_flag'] == 'P')
    # setup filter for 'OTM' calls
    is_otm_call = (df['strike_price'] >= K0) & (df['cp_flag'] == 'C')

    # only keep 'OTM' puts and calls
    df = df[ is_otm_call | is_otm_put ]

    # calculate the delta_K values
    # for the smallest strike we use:       delta_K_1 := K_2 - K_1
    # for the largest strike we use:        delta_K_n := K_n - K_{n-1}
    # for all strikes inbetween we use:     delta_K_s := (K_{s+1} - K_{s-1}) / 2
    df['delta_K'] = ( df['strike_price'].shift(-1) - df['strike_price'].shift(1) ) / 2
    df['delta_K'].iloc[0] = df['strike_price'].iloc[1] - df['strike_price'].iloc[0]
    df['delta_K'].iloc[-1] = df['strike_price'].iloc[-1] - df['strike_price'].iloc[-2]

    # calculate sum part
    tau = df['time_to_mat'].iloc[0] / 250  # annualized time to maturity (same for all contracts in df)
    sum_part = np.sum( df['price'] * df['delta_K']/(df['strike_price']*df['strike_price']) ) * np.exp(r * tau)

    # calculate implied variance
    impl_var = (1/tau) * ( 2*sum_part - (F/K0 - 1)**2 )

    # return implied volatility
    return np.sqrt(impl_var)


# By Bakshi et al. (2003)
def modelfree_ivol(smile, spot, r):
    df = smile.copy()
    df = df.sort_values('strike_price')
    # prepare integration
    K = df['strike_price']
    df['V:H_SS'] = 2 * ( 1  - np.log( K / spot ) ) / K**2       # integrand of V contract
    df['W:H_SS'] = ( 6 * np.log(K / spot) - 3 * np.log(K / spot)**2 ) / K ** 2      # integrand of W
    df['X:H_SS'] = ( 12 * np.log(K / spot)**2 - 4 * np.log(K / spot)**3 ) / K ** 2      # integrand of X
    # estimate integral with simple block approximation
    df['d_K'] = df['strike_price'].diff(1).shift(-1)
    V = np.sum( df['V:H_SS'] * df['price'] * df['d_K'] )       # price for V(t, tau) contract
    W = np.sum( df['W:H_SS'] * df['price'] * df['d_K'] )      # price for W(t, tau) contract
    X = np.sum( df['X:H_SS'] * df['price'] * df['d_K'] )      # price for X(t, tau) contract
    # also estimate integral using the trapezoidal rule
    V_trapz = np.trapz(df['V:H_SS'] * df['price'], K)       # price for V(t, tau) contract
    W_trapz = np.trapz(df['W:H_SS'] * df['price'], K)       # price for W(t, tau) contract
    X_trapz = np.trapz(df['X:H_SS'] * df['price'], K)       # price for X(t, tau) contract

    # calculate helper variables
    tau = df['time_to_mat'].iloc[0] / 250      # annualized time to maturity (same for all contracts in df)
    ertau = np.exp(r*tau)       # compound factor: e^(r*tau)

    def compute_higher_moments(V_, W_, X_):
        # calculate helper variable (mu)
        mu_ = ertau - 1 - (ertau / 2) * V_ - (ertau / 6) * W_ - (ertau / 24) * X_
        # calculate model-free variance
        var_ = ertau*V_ - mu_**2
        # calculate model-free skew and kurtosis
        a_skew_ = ertau*W_ - 3*mu_*ertau*V_ + 2*(mu_**3)
        a_kurt_ = ertau*X_ - 4*mu_*ertau*W_ + 6*ertau*(mu_**2)*V_ - 3*(mu_**4)
        b_ = ertau*V_ - mu_**2
        b_skew_ = np.power(b_, 1.5)
        b_kurt_ = b_**2
        skew_ = a_skew_/b_skew_
        kurt_ = a_kurt_/b_kurt_
        # annualize approximated variance, skew and kurtosis
        annualize_factor = 1/tau        # unit of tau is years
        var_ = var_ * annualize_factor
        skew_ = skew_ / np.sqrt(annualize_factor)   # see cumulants
        kurt_ = kurt_ / annualize_factor    # see cumulants

        return var_, skew_, kurt_

    var, skew, kurt = compute_higher_moments(V, W, X)
    var_trapz, skew_trapz, kurt_trapz = compute_higher_moments(V_trapz, W_trapz, X_trapz)

    # we return volatility, skew and kurtosis (from block and trapezoidal integral approximation)
    return (np.sqrt(var), skew, kurt), (np.sqrt(var_trapz), skew_trapz, kurt_trapz)


def custom_fit(x_vals, y_vals, centre):
    # The structure of the function permits good initial guesses for (a, b, c, d) to be found directly by
    # fitting the left or right part of the curve using least squares
    def initial_fit(x, y, x0, y0, side='left'):
        assert side in {'left', 'right'}
        # shift data
        x_shift = x - x0
        if side == 'left':
            idx_slice = x_shift < 0
        else:
            idx_slice = x_shift >= 0
        x_slice, y_slice = x_shift.loc[idx_slice], y.loc[idx_slice]
        y_slice -= y0
        # perform least squares of corresponding side (left/right)
        X = np.vstack([x_slice ** 2, x_slice]).T  # design matrix
        (p1, p2), residuals, rank, s = np.linalg.lstsq(X, y_slice, rcond=None)
        return p1, p2

    # loss function is needed for optimisation -- MSE
    def loss(params, x_vals, y_vals):
        predictions = [custom_function(x, *params) for x in x_vals]
        return np.average((predictions - y_vals) ** 2)

    # perform optimisation 3 times to ensure best result amongst 3 different (random) initialisations
    error_list = []
    param_list = []
    for i in range(1):
        # perform optimisation
        # first, set initial parameters and bounds
        idx_of_roughly_1_in_xvals = (x_vals - centre).abs().idxmin()
        centre_y = y_vals.loc[idx_of_roughly_1_in_xvals]
        a_initial, b_initial = initial_fit(x_vals, y_vals, centre, centre_y, side='left')
        c_initial, d_initial = initial_fit(x_vals, y_vals, centre, centre_y, side='right')
        # midpoint and midpoint_y are x and y translations, respectively
        initial_params = [a_initial, b_initial, c_initial, d_initial, centre, centre_y]
        # set random initialisation for smoothing kernel size (alpha)
        initial_params.append(np.random.uniform(0.1, 1))
        # set bounds
        bounds = [(a_initial - 3, a_initial + 3),
                  (b_initial - 3, b_initial + 3),
                  (c_initial - 3, c_initial + 3),
                  (d_initial - 3, d_initial + 3),
                  (centre - 0.2, centre + 0.2),
                  (centre_y - 0.2, centre_y + 0.2),
                  (0.1, 1.0)]
        # second, perfom minimisation
        t_minimize_0 = time.time()
        result = minimize(loss, initial_params, args=(x_vals, y_vals), bounds=bounds)
        t_minimize_1 = time.time()
        optimized_params = result.x
        # append optimisation results to error and param list
        error_list.append(loss(optimized_params, x_vals, y_vals))
        param_list.append(optimized_params)

    # choose optimal parameters
    error_vec = np.array(error_list)
    best_params = param_list[np.argmin(error_vec)]

    # retrieve error (MSE)
    error = np.min(error_vec)

    # prepare final parameters vector
    x_min, x_max = min(x_vals), max(x_vals)
    # parameters vector is arranged as: [x_min, x_max, param_0, param_1, param_2, ...]
    all_param = np.concatenate([np.array([x_min, x_max]), best_params])

    # define fitted model
    def fitted_model(x):
        return custom_function(x, *best_params)

    # calculate integrated difference to linear interpolation
    x_line = np.linspace(min(x_vals), max(x_vals), 1000)
    idx_sorted = np.argsort(x_vals)
    y_interp_lin = np.interp(x_line, x_vals.iloc[idx_sorted], y_vals.iloc[idx_sorted])
    y_interp_spl = CubicSpline(x_vals.iloc[idx_sorted], y_vals.iloc[idx_sorted])(x_line)
    y_model = fitted_model(x_line)
    error_interp_lin = np.mean( (y_interp_lin-y_model)**2 )     # ≈ integral of squared diff. to linear interpolation
    error_interp_spl = np.mean((y_interp_spl - y_model) ** 2)

    return fitted_model, all_param, (error, error_interp_lin, error_interp_spl)


def spline_fit(x_vals, y_vals, centre, n_knots=7):
    model = get_natural_cubic_spline_model(x_vals, y_vals,
                                           minval=min(x_vals), maxval=max(x_vals),
                                           n_knots=n_knots)
    # calculate error (MSE)
    y_fit = model.predict(x_vals)
    error = np.average((y_fit-y_vals)**2)

    # prepare final parameters vector
    coef = model.named_steps.regression.coef_
    intercept = model.named_steps.regression.intercept_
    x_min, x_max = min(x_vals), max(x_vals)
    # parameters vector is arranged as: [x_min, x_max, intercept (=coef_0), coef_1, coef_2, ...]
    all_param = np.concatenate( [np.array([x_min, x_max, intercept]), coef] )

    # calculate integrated difference to linear interpolation
    x_line = np.linspace(min(x_vals), max(x_vals), 1000)
    idx_sorted = np.argsort(x_vals)
    y_interp_lin = np.interp(x_line, x_vals.iloc[idx_sorted], y_vals.iloc[idx_sorted])
    y_interp_spl = CubicSpline(x_vals.iloc[idx_sorted], y_vals.iloc[idx_sorted])(x_line)
    y_model = model.predict(x_line)
    error_interp_lin = np.mean( (y_interp_lin-y_model)**2 )     # ≈ integral of squared diff. to linear interpolation
    error_interp_spl = np.mean((y_interp_spl - y_model) ** 2)   # ≈ integral of squared diff. to spline interpolation

    return model.predict, all_param, (error, error_interp_lin, error_interp_spl)



def get_riskfree_rate_TB3MS(date):
    # need dataframe 'stock_price_df' in base directory with column 'TB3MS', indexed by date
    assert os.path.isfile('additional_dfs/riskfree_rates_TB3MS.parquet')  # provide file
    riskfree_rate_df = pd.read_parquet('additional_dfs/riskfree_rates_TB3MS.parquet')
    r = riskfree_rate_df.loc[date].TB3MS
    return r


def get_riskfree_rate(date, days, zc_df=None):
    if zc_df is None:
        assert os.path.isfile('additional_dfs/zero_curve.parquet')  # provide file
        zc_df = pd.read_parquet('additional_dfs/zero_curve.parquet')
    if date not in zc_df.index:
        return np.nan   # no data available for that day
    rates = zc_df.loc[date]
    rates_l = rates[rates['days'] <= days]  # rates with shorter maturity
    rates_u = rates[rates['days'] >= days]  # rates with longer maturity

    if len(rates_l)==0:
        return rates.iloc[0].rate/100   # if 'days' smaller than available => use rate for longest maturity
    if len(rates_u)==0:
        return rates.iloc[-1].rate/100  # if 'days' larger than available => use rate for shortest maturity

    # otherwise interpolate rate between two closest dates available
    rate_l = rates_l.iloc[-1]
    rate_u = rates_u.iloc[0]
    if rate_u.days==rate_l.days:
        return rate_l.rate      # in case we have an exact match
    # f(x) = [(x-a)*B + (b-x)*A]/(b-a)
    r = ( (days - rate_l.days)*rate_u.rate + (rate_u.days - days)*rate_l.rate ) / ( rate_u.days - rate_l.days )
    return r/100


def get_spot_price(ticker, date):
    # need dataframe 'stock_price_df' in base directory with columns ['date', 'ticker', 'price']
    assert os.path.isfile('additional_dfs/stock_price_df.parquet')     # provide file
    stock_price_df = pd.read_parquet('additional_dfs/stock_price_df.parquet')
    filter_res = stock_price_df[ (stock_price_df['ticker']==ticker) & (stock_price_df['date']==date) ]
    if len(filter_res)>0:
        return filter_res.iloc[0].price     # return price for ticker and date
    return None     # can't find stock price


def process_option_group(df, date, ticker, min_mat, max_mat, num_mat):
    assert min_mat < 21 < max_mat       # number of trading days in month ≈ 250/12 ≈ 21
    # get spot price
    spot = get_spot_price(ticker, date)
    # if spot == None --> cannot find spot price
    if not spot:
        print(f'No spot price available for: [{date}, {ticker}]')
        return None

    ticker_df = df[(df['ticker'] == ticker)
                   & (df['date'] == date)
                   & (df['time_to_mat'] < max_mat)
                   & (df['time_to_mat'] >= min_mat)]
    t_to_mat_all = ticker_df['time_to_mat'].unique()
    # we want to sort the curves based on proximity to 21 trading days
    idx_t_to_mat_all_sorted = np.argsort(np.abs(t_to_mat_all - 21))
    t_to_mat_all_sorted = t_to_mat_all[idx_t_to_mat_all_sorted]
    # the curves array will be populated later
    smile_data = []     # will be populated with processed data of smiles
    # we only loop through the first 'num_mat' categories (the 'num_mat' maturities closest to 21)

    # time measurements

    for crt_t_to_mat in t_to_mat_all_sorted:

        t_prep_stuff_0 = time.time()
        if len(smile_data) == num_mat:
            break  # only keep the best 'num_mat' curves

        crt_smile = ticker_df[ticker_df['time_to_mat'] == crt_t_to_mat]

        days_to_mat = crt_smile.iloc[0]['time_to_mat_calendar']   # calendar days to maturity
        # r = get_riskfree_rate_TB3MS(date)     # risk free rate from T-Bill rates (3 months)
        r = get_riskfree_rate(date, days_to_mat)  # risk free rate (zero curve file, optionmetrics), annualized
        results_smile = {}  # will be populated later

        # perform calculations with curve
        # drop some of the data points
        clean_smile_data, feature_dict = clean_option_data_Prices(crt_smile, spot)
        if len(clean_smile_data) < 3:
            break   # if too little clean data, skip this curve
        results_smile.update(feature_dict)  # add features extracted during cleaning

        # prepare variables for fitting
        moneyness = spot / clean_smile_data['strike_price']
        neg_log_moneyness = -np.log( moneyness )
        impl_vol = clean_smile_data['impl_volatility']
        prices = clean_smile_data['price']

        t_custom_fit_0 = time.time()
        # # custom fit
        model_custom, param, errors = custom_fit(moneyness, impl_vol, centre=1)    # moneyness vs implied
        results_smile['param_custom_moneyness_impl'] = param
        error_MSE, error_interp_lin, error_interp_spl = errors
        results_smile['error_MSE_model_custom_moneyness_impl'] = error_MSE
        results_smile['error_interp_lin_model_custom_moneyness_impl'] = error_interp_lin
        results_smile['error_interp_spl_model_custom_moneyness_impl'] = error_interp_spl

        _, param, errors = custom_fit(neg_log_moneyness, impl_vol, centre=0)    # -log(moneyness) vs implied
        results_smile['param_custom_nlmoneyness_impl'] = param
        error_MSE, error_interp_lin, error_interp_spl = errors
        results_smile['error_MSE_model_custom_nlmoneyness_impl'] = error_MSE
        results_smile['error_interp_lin_model_custom_nlmoneyness_impl'] = error_interp_lin
        results_smile['error_interp_spl_model_custom_nlmoneyness_impl'] = error_interp_spl
        # temporarily commented out due to high computational cost
        #_, param, error, t_min_3 = custom_fit(moneyness, prices, centre=1)  # moneyness vs price
        #results_smile['param_custom_moneyness_price'] = param
        #results_smile['error_model_custom_moneyness_price'] = error
        #_, param, error, t_min_4 = custom_fit(neg_log_moneyness, prices, centre=0)  # -log(moneyness) vs price
        #results_smile['param_custom_nlmoneyness_price'] = param
        #results_smile['error_model_custom_nlmoneyness_price'] = error

        # cubic spline fit
        model_spline, param, errors = spline_fit(moneyness, impl_vol, centre=1)    # moneyness vs implied
        results_smile['param_spline_moneyness_impl'] = param
        error_MSE, error_interp_lin, error_interp_spl = errors
        results_smile['error_MSE_model_spline_moneyness_impl'] = error_MSE
        results_smile['error_interp_lin_model_spline_moneyness_impl'] = error_interp_lin
        results_smile['error_interp_spl_model_spline_moneyness_impl'] = error_interp_spl
        _, param, errors = spline_fit(neg_log_moneyness, impl_vol, centre=0)    # -log(moneyness) vs implied
        results_smile['param_spline_nlmoneyness_impl'] = param
        error_MSE, error_interp_lin, error_interp_spl = errors
        results_smile['error_MSE_model_spline_nlmoneyness_impl'] = error_MSE
        results_smile['error_interp_lin_model_spline_nlmoneyness_impl'] = error_interp_lin
        results_smile['error_interp_spl_model_spline_nlmoneyness_impl'] = error_interp_spl
        _, param, errors = spline_fit(moneyness, prices, centre=1)  # moneyness vs price
        results_smile['param_spline_moneyness_price'] = param
        error_MSE, error_interp_lin, error_interp_spl = errors
        results_smile['error_MSE_model_spline_moneyness_price'] = error_MSE
        results_smile['error_interp_lin_model_spline_moneyness_price'] = error_interp_lin
        results_smile['error_interp_spl_model_spline_moneyness_price'] = error_interp_spl
        _, param, errors = spline_fit(neg_log_moneyness, prices, centre=0)  # -log(moneyness) vs price
        results_smile['param_spline_nlmoneyness_price'] = param
        error_MSE, error_interp_lin, error_interp_spl = errors
        results_smile['error_MSE_model_spline_nlmoneyness_price'] = error_MSE
        results_smile['error_interp_lin_model_spline_nlmoneyness_price'] = error_interp_lin
        results_smile['error_interp_spl_model_spline_nlmoneyness_price'] = error_interp_spl

        t_additional_features_0 = time.time()
        # calculate some additional features
        # ATM implied volatility (from both models: model_spline & model_custom)
        results_smile['atm_ivol_custom'] = model_custom(1.)
        results_smile['atm_ivol_spline'] = model_spline(np.ones(1))[0]   # input needs to be np array --> returns np array
        # ATM slope (from both models: model_spline & model_custom)
        eps = 1e-4
        results_smile['atm_slope_custom'] = (model_custom(1.+eps)-model_custom(1.)) / eps
        results_smile['atm_slope_spline'] = (model_spline(np.ones(1)+eps)[0] - model_spline(np.ones(1))[0]) / eps
        # model-free implied volatility, skew and kurtosis
        mf_ivol_raw_moments, mf_ivol_raw_moments_trapz = modelfree_ivol(clean_smile_data, spot, r)
        (results_smile['mf_ivol_raw_vol'],
         results_smile['mf_ivol_raw_skew'],
         results_smile['mf_ivol_raw_kurt']) = mf_ivol_raw_moments
        (results_smile['mf_ivol_raw_vol_trapz'],
         results_smile['mf_ivol_raw_skew_trapz'],
         results_smile['mf_ivol_raw_kurt_trapz']) = mf_ivol_raw_moments_trapz
        # model-free implied volatility from CBOE (VIX algorithm)
        results_smile['mf_ivol_cboe'] = modelfree_ivol_cboe(clean_smile_data, r)

        puts = clean_smile_data[clean_smile_data['cp_flag']=='P']
        calls = clean_smile_data[clean_smile_data['cp_flag']=='C']
        # option volume and number of contracts
        results_smile['number_contracts_put'] = len(puts)
        results_smile['number_contracts_call'] = len(calls)
        results_smile['total_volume_put'] = sum( puts.volume )
        results_smile['total_volume_call'] = sum( calls.volume )

        # average bid-ask spread
        results_smile['ba_spread_avg'] = np.average(clean_smile_data['best_offer'] - clean_smile_data['best_bid'])
        results_smile['ba_spread_avg_put'] = np.average(puts['best_offer'] - puts['best_bid'])
        results_smile['ba_spread_avg_call'] = np.average(calls['best_offer'] - calls['best_bid'])
        # median bid-ask spread
        results_smile['ba_spread_median'] = np.median(clean_smile_data['best_offer'] - clean_smile_data['best_bid'])
        results_smile['ba_spread_median_put'] = np.median(puts['best_offer'] - puts['best_bid'])
        results_smile['ba_spread_median_call'] = np.median(calls['best_offer'] - calls['best_bid'])

        # time to maturity (trading days)
        results_smile['time_to_mat'] = crt_t_to_mat

        # add results_smile to results
        smile_data.append(results_smile)

    # change keys in dictionary to capture which curve it is
    for i, results_smile in enumerate(smile_data):
        smile_data[i] = {f'smile_{i}::{key}': value for key, value in results_smile.items()}

    # now concatenate all dictionaries in smile_data --> return
    all_dict = {}
    for d in smile_data:
        all_dict.update(d)

    all_dict['date'] = date
    all_dict['ticker'] = ticker

    if all_dict:
        return all_dict     # return data gathered

    return None     # no data could be gathered


def extract_features_from_price_data_Prices(csv_file_path, parquet_file_path, column_types, date_cols,
                                            min_mat, max_mat, num_mat, suffix_range, chunksize=1e6):
    # check whether data is available in parquets
    # if not, run conversion first
    if (( not os.path.isdir(parquet_file_path) ) or
            ( len(get_files_in_folder(parquet_file_path, 'parquet')) == 0 )):
        print('######################')
        print('running conversion to parquet...')
        print('######################')
        convert_csv_to_parquets_Prices(csv_file_path, column_types, date_cols,
                                       output_folder=parquet_file_path, chunksize=chunksize)
        print('######################')
        print('done with conversion to parquet...')
        print('######################')

    parquet_chunks = get_files_in_folder(parquet_file_path, 'parquet')
    # filter for the range given by suffix_range (if suffix_range contains elements)
    if len(suffix_range)==2:
        a, b = suffix_range
        parquet_chunks = [ fn for fn in parquet_chunks if int(fn[:-8].split('_')[-1]) in range(a,b) ]

    # sort filenames lexicographically so that the database is iterated in correct order
    # not necessary but seems cleaner to me
    parquet_chunks.sort()

    t_temp_0 = time.time()
    print('######################')
    print('processing the chunks...')
    print('######################')
    results = []  # for collecting the results of processing the data
    for i, filename in enumerate(parquet_chunks):
        print(f'processing chunk {i+1} of {len(parquet_chunks)}')
        t_before_open = time.time()
        df_chunk = pd.read_parquet(filename)
        t_after_open = time.time()
        print(f"Time to open parquet: {t_after_open - t_before_open}")

        # group the data in df_chunk by 'date' and 'ticker'
        # => each group contains all available option data for specific ticker on specific date
        option_groups = df_chunk.groupby(['date', 'ticker'])
        print(f"{len(option_groups)} groups in this chunk file")
        t_process_total = 0
        for group_idx, ((crt_date, crt_ticker), crt_group_df) in enumerate(option_groups):

            t_process_0 = time.time()
            result_group = process_option_group(crt_group_df, crt_date, crt_ticker,
                                                min_mat, max_mat, num_mat)      # represents one row
            t_process_1 = time.time()
            t_process_total += (t_process_1 - t_process_0)
            t_mean = t_process_total/(group_idx + 1)
            print(f"Group {group_idx+1}/{len(option_groups)}: {(t_process_1 - t_process_0):.3f} s")
            print(f"mean time per group: {t_mean:.3f} s")
            if result_group:
                results.append(result_group)        # 'process_option_group()' might return None

    # all the data has been proessed
    # now we need to convert the result list into a usable dataframe
    df_results = pd.DataFrame(results)
    df_results_pivot = df_results.pivot(columns='ticker', index='date')
    df_results_pivot = swap_levels(df_results_pivot)

    if len(suffix_range) == 2:
        a, b = suffix_range
        df_results_pivot.to_parquet(f'options_prices_features_{a}_{b}.parquet')     # include file range if available
    else:
        df_results_pivot.to_parquet(f'options_prices_features.parquet')

    t_temp_1 = time.time()
    print('######################')
    print('done with processing the chunks!')
    print(f"That took {int(t_temp_1 - t_temp_0)} seconds")
    print('######################')

########################################################################################################################


# assumes that index of df is reset before calling the function
def split_chunk(df):
    # We want to keep data of the same day + the same ticker in the same chunk
    # These groups are cut off by default by using chunks
    # We want to carry the last part of each chunk [the cut-off group] to the next chunk
    # => reconstruct the group with the beginning of the next chunk and store it in the next chunk
    # This way, each group is contained in exactly one chunk
    # => Later on, when we need data on the same day & ticker, we don't have to search in other chunks!

    # This function helps by separating the "last part" from the rest of "df"
    # 1) find idx where last group of the chunk begins
    last_ticker = df.iloc[-1]['ticker']
    last_date = df.iloc[-1]['date']
    split_idx = df[(df['date'] == last_date) & (df['ticker'] == last_ticker)].iloc[0].name
    # 2) split chunk at this index
    previous_df = df.iloc[split_idx:]
    df = df.iloc[:split_idx]

    return previous_df, df


def convert_csv_to_parquets_Prices(csv_file_path, column_types, date_cols, output_folder, chunksize=1e6):
    # create directory for parquet files if it does not exist already
    if not os.path.isdir(f'{output_folder}/'):
        os.makedirs(f'{output_folder}/')

    assert len(get_files_in_folder(f'{output_folder}/', 'parquet'))==0   # clear old data in directory first
    # forward price file from option metrics (only kept columns ticker, date, expiration)
    # renamed column 'expiration' -> 'exdate' for cleaner merging
    assert os.path.isfile('additional_dfs/forward_prices.parquet')     # provide file
    fw_df = pd.read_parquet('additional_dfs/forward_prices.parquet')

    previous_df = pd.DataFrame()
    chunks = pd.read_csv(csv_file_path,
                         chunksize=int(chunksize),
                         parse_dates=date_cols,
                         usecols=[*list(column_types.keys()), *date_cols],
                         dtype=column_types,
                         date_format='%Y-%m-%d')
    i = 1
    for chunk_df in chunks:
        print(f"Starting with chunk {i}...")
        t_temp_0 = time.time()
        # prepare chunk
        chunk_df = chunk_df.dropna()
        # calculate times to maturity
        chunk_df['time_to_mat_calendar'] = chunk_df.exdate - chunk_df.date        # time to maturity
        chunk_df['time_to_mat_calendar'] = chunk_df['time_to_mat_calendar'].dt.days  # days to maturity
        chunk_df['time_to_mat'] = chunk_df['time_to_mat_calendar'] * (250/365)    # ≈ trading days to maturity

        chunk_df['strike_price'] = chunk_df['strike_price']/1000
        # concatenate with carried dataframe from last chunk (previous_df)
        df = pd.concat([previous_df, chunk_df], axis='rows')
        df = df.reset_index(drop=True)

        # calculate part of 'df' to carry (previous_df)
        previous_df, df = split_chunk(df)

        # merge df with fw_df to attach forward price column
        df = pd.merge(left=df, right=fw_df, on=['ticker', 'date', 'exdate'], how='left')

        # save df to disk
        df.to_parquet(f'{output_folder}/df_{i}.parquet')

        t_temp_1 = time.time()
        print(f"That took {(t_temp_1 - t_temp_0):.3f} seconds")
        i += 1

    # At the end, we need to save the carried df (previous_df) to disk
    previous_df.to_parquet(f'{output_folder}/df_{i}.parquet')

    # This last bit is to adjust the naming --> will allow to sort files lexicographically
    # gather all names of dataframes stored and the number of digits of the longest suffix
    df_names = get_files_in_folder(f'{output_folder}/', 'parquet')
    df_name_lengths = [len(s) for s in df_names]
    max_len = max(df_name_lengths)
    # subtract len(output_folder) and 12 [12=1+3+8; 1=len('/') & 3=len('df_') & 8=len('.parquet')]
    # => get number of digits of longest suffix
    max_len -= ( len(output_folder) + 12)
    # We want all suffixes to be of consistent length ('max_len') so that we can sort them lexicographically
    for df_name in df_names:
        number = df_name.split('_')[-1].split('.')[0]
        new_number = number.zfill(max_len)
        os.rename(f'{output_folder}/{df_name}', f'{output_folder}/df_{new_number}.parquet')




if __name__ == "__main__":

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    print(f"Start time: {formatted_time}")
    # for submission on HTC cluster
    suffix_range = []
    parser = argparse.ArgumentParser(description='Specify the indices of the parquet files to consider (e.g. "1:20")')
    parser.add_argument('index_range', type=str, help='The indices to consider.')
    my_arg = parser.parse_args().index_range
    suffix_range = [int(s) for s in my_arg.split(':')]
    # so that I do not have to copy the parquet files
    parquet_file_path = os.path.expandvars('$DATA/Option_Features/Option_Prices_Parquet/')

    # Preparation
    ####################################################################################################################
    chunk_size = 1e6    # parse chunks of 1 Million rows
    csv_path_option_prices = "lyxrbbknrgqdt41z.csv"     # Option_Price file
    column_types = {'secid': np.int64,
                    'ticker': str,
                    'strike_price': np.int64,
                    'best_bid': np.float64,
                    'best_offer': np.float64,
                    'volume': np.int64,
                    'impl_volatility': np.float64,
                    'delta': np.float64,
                    'gamma': np.float64,
                    'vega': np.float64,
                    'theta': np.float64,
                    'exercise_style': str,
                    'cp_flag': str}

    date_cols = ['date', 'exdate']
    ####################################################################################################################

    # start feature extraction in chunks
    t1 = time.time()
    # 'min_mat' and 'max_mat' are given in trading days
    extract_features_from_price_data_Prices(csv_path_option_prices, parquet_file_path, column_types, date_cols,
                                            min_mat=10, max_mat=50, num_mat=1,
                                            suffix_range=suffix_range)
    t2 = time.time()
    print(f'Feature extraction : {int(t2 - t1)} seconds')
