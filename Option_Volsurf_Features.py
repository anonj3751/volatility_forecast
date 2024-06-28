import numpy as np
import pandas as pd
import os
import time
from Option_Price_Features import split_chunk
from helper import get_files_in_folder
from helper import swap_levels
from Option_Price_Features import get_spot_price
from Option_Price_Features import get_riskfree_rate
from Option_Price_Features import modelfree_ivol
from Option_Price_Features import spline_fit
from Option_Price_Features import custom_fit
import argparse


def clean_option_data_VolSurface(df, spot):
    feature_dict = {}

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

    return df, feature_dict


def process_option_group_VolSurface( df, date, ticker, ):
    # get spot price
    spot = get_spot_price(ticker, date)
    # if spot == None --> cannot find spot price
    if not spot:
        print(f'No spot price available for: [{date}, {ticker}]')
        return None

    ticker_df = df[(df['ticker'] == ticker)
                   & (df['date'] == date)] # dropping all data for maturities =|= 30 is done in csv->parquet conversion

    days_to_mat = 30   # calendar days to maturity
    r = get_riskfree_rate(date, days_to_mat)  # risk free rate (zero curve file, optionmetrics), annualized
    results_smile = {}  # will be populated later

    # perform calculations with curve
    # drop some of the data points
    clean_smile_data, feature_dict = clean_option_data_VolSurface(ticker_df, spot)
    if len(clean_smile_data) < 3:
        return None  # if too little clean data, skip this curve
    results_smile.update(feature_dict)  # add features extracted during cleaning

    # prepare variables for fitting
    moneyness = spot / clean_smile_data['strike_price']
    neg_log_moneyness = -np.log( moneyness )
    impl_vol = clean_smile_data['impl_volatility']
    prices = clean_smile_data['price']

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

    t_spline_fit_0 = time.time()
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

    puts = clean_smile_data[clean_smile_data['cp_flag']=='P']
    calls = clean_smile_data[clean_smile_data['cp_flag']=='C']
    # option volume and number of contracts
    results_smile['number_contracts_put'] = len(puts)
    results_smile['number_contracts_call'] = len(calls)

    # change keys in dictionary to capture which curve it is
    results_smile = {f'smile_volsurf::{key}': value for key, value in results_smile.items()}

    # now concatenate all dictionaries in smile_data --> return
    results_smile['date'] = date
    results_smile['ticker'] = ticker

    if results_smile:
        return results_smile     # return data gathered

    return None     # no data could be gathered


def extract_features_from_price_data_VolSurfaces(csv_path_volsurf, parquet_file_path, column_types,
                                                 date_cols, suffix_range, chunksize=1e6):
    # check whether data is available in parquets
    # if not, run conversion first
    if (( not os.path.isdir(parquet_file_path) ) or
            (len(get_files_in_folder(parquet_file_path, 'parquet')) == 0)):
        print('######################')
        print('running conversion to parquet...')
        print('######################')
        convert_csv_to_parquets_VolSurface(csv_path_volsurf, column_types, date_cols,
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
        print(f'processing chunk {i+1} of {len(parquet_chunks)}: {filename}')
        df_chunk = pd.read_parquet(filename)

        # group the data in df_chunk by 'date' and 'ticker'
        # => each group contains all available option data for specific ticker on specific date
        option_groups = df_chunk.groupby(['date', 'ticker'])
        print(f"{len(option_groups)} groups in this chunk file")
        t_process_total = 0
        for group_idx, ((crt_date, crt_ticker), crt_group_df) in enumerate(option_groups):
            t_process_0 = time.time()
            result_group = process_option_group_VolSurface(crt_group_df, crt_date, crt_ticker)  # represents one row
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
    df_results_pivot.to_parquet('options_volsurf_features.parquet')

    if len(suffix_range) == 2:
        a, b = suffix_range
        df_results_pivot.to_parquet(f'options_volsurf_features_{a}_{b}.parquet')     # include file range if available
    else:
        df_results_pivot.to_parquet(f'options_volsurf_features.parquet')

    t_temp_1 = time.time()
    print('######################')
    print('done with processing the chunks!')
    print(f"That took {int(t_temp_1 - t_temp_0)} seconds")
    print('######################')


# function for converting csv to parquet files --> specialized for VolatilitySurface file
# see also convert_csv_to_parquets_Prices function
def convert_csv_to_parquets_VolSurface(csv_file_path, column_types, date_cols, output_folder,
                                       filter_maturity=True, chunksize=1e6):
    # create directory for parquet files if it does not exist already
    if not os.path.isdir(f'{output_folder}/'):
        os.makedirs(f'{output_folder}/')

    current_files_in_dir = get_files_in_folder(f'{output_folder}/', 'parquet')
    assert len(current_files_in_dir)==0   # clear old data in directory first
    # forward price file from option metrics (only kept columns ticker, date, expiration)
    # renamed column 'expiration' -> 'exdate' for cleaner merging

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
        # rename columns to match Option_Price_Features
        chunk_df = chunk_df.rename(columns={'impl_strike': 'strike_price',
                                            'days': 'time_to_mat_calendar',
                                            'impl_premium': 'price'
                                            })

        if filter_maturity:
            chunk_df = chunk_df[ chunk_df['time_to_mat_calendar']==30 ]     # we are only interested in 30 days to expiry

        chunk_df['time_to_mat'] = chunk_df['time_to_mat_calendar']*(250/365)    # approximate trading days
        # concatenate with carried dataframe from last chunk (previous_df)
        df = pd.concat([previous_df, chunk_df], axis='rows')
        df = df.reset_index(drop=True)

        # calculate part of 'df' to carry (previous_df)
        previous_df, df = split_chunk(df)

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
    # subtract len(output_folder) + 12 [12=1+3+8; 1=('/') & 3=len('df_') & 8=len('.parquet')]
    # => get number of digits of longest suffix
    max_len -= ( len(output_folder) + 12)
    # We want all suffixes to be of consistent length ('max_len') so that we can sort them lexicographically
    for df_name in df_names:
        number = df_name.split('_')[-1].split('.')[0]
        new_number = number.zfill(max_len)
        os.rename(f'{df_name}', f'{output_folder}/df_{new_number}.parquet')


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
    # so that I do not have to copy the parquet files (for ARC)
    parquet_file_path = os.path.expandvars('$DATA/Option_Features/Option_VolSurface_Parquet/')

    column_types = {'days': np.int64,
                    'delta': np.float64,
                    'impl_volatility': np.float64,
                    'impl_strike': np.float64,
                    'impl_premium': np.float64,
                    'cp_flag': str,
                    'ticker': str}
    date_cols = ['date']

    csv_path_volsurf = ("zued4aezou2lhwsj.csv")     # path to Volatility_Surface file
    df_volsurf = pd.read_csv(csv_path_volsurf,
                             nrows=1000000,
                             parse_dates=date_cols,
                             usecols=[*list(column_types.keys()), *date_cols],
                             dtype=column_types,
                             date_format='%Y-%m-%d')
    t1 = time.time()
    extract_features_from_price_data_VolSurfaces(csv_path_volsurf, parquet_file_path,
                                                 column_types, date_cols, suffix_range, chunksize=1e6)
    t2 = time.time()
    print(f'Feature extraction : {int(t2 - t1)} seconds')
