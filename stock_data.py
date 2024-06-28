import pandas as pd
import numpy as np
from plotting import plot_data_availability, plot_two_data_availabilities
from helper import swap_levels
from helper import get_files_in_folder
from scipy.stats.mstats import winsorize


########################################################################################################################
# function for importing and preparing stock data
########################################################################################################################
def read_tickers(ticker_filepath):
    with open(ticker_filepath, 'r') as ticker_file:
        tickers = ticker_file.read()
    # open as list (split by linebreak) and strip leading/trailing spaces
    tickers = tickers.split('\n')
    tickers = set([ticker.strip() for ticker in tickers])
    return tickers


def get_finratio_firmlevel(relevant_tickers):
    filepath = ('path_to_financial_ratios')     # provide file
    column_types = {
        'CAPEI': np.float64,
        'GProf': np.float64,
        'PEG_trailing': np.float64,
        'TICKER': str,
        'accrual': np.float64,
        'adv_sale': np.float64,
        'aftret_eq': np.float64,
        'aftret_equity': np.float64,
        'aftret_invcapx': np.float64,
        'at_turn': np.float64,
        'bm': np.float64,
        'capital_ratio': np.float64,
        'cash_conversion': np.float64,
        'cash_debt': np.float64,
        'cash_lt': np.float64,
        'cash_ratio': np.float64,
        'cfm': np.float64,
        'curr_debt': np.float64,
        'curr_ratio': np.float64,
        'de_ratio': np.float64,
        'debt_assets': np.float64,
        'debt_at': np.float64,
        'debt_capital': np.float64,
        'debt_ebitda': np.float64,
        'debt_invcap': np.float64,
        'divyield': str,
        'dltt_be': np.float64,
        'dpr': np.float64,
        'efftax': np.float64,
        'equity_invcap': np.float64,
        'evm': np.float64,
        'fcf_ocf': np.float64,
        'gpm': np.float64,
        'int_debt': np.float64,
        'int_totdebt': np.float64,
        'intcov': np.float64,
        'intcov_ratio': np.float64,
        'inv_turn': np.float64,
        'invt_act': np.float64,
        'lt_debt': np.float64,
        'lt_ppent': np.float64,
        'npm': np.float64,
        'ocf_lct': np.float64,
        'opmad': np.float64,
        'opmbd': np.float64,
        'pay_turn': np.float64,
        'pcf': np.float64,
        'pe_exi': np.float64,
        'pe_inc': np.float64,
        'pe_op_basic': np.float64,
        'pe_op_dil': np.float64,
        'pretret_earnat': np.float64,
        'pretret_noa': np.float64,
        'profit_lct': np.float64,
        'ps': np.float64,
        'ptb': np.float64,
        'ptpm': np.float64,
        'quick_ratio': np.float64,
        'rd_sale': np.float64,
        'rect_act': np.float64,
        'rect_turn': np.float64,
        'roa': np.float64,
        'roce': np.float64,
        'roe': np.float64,
        'sale_equity': np.float64,
        'sale_invcap': np.float64,
        'sale_nwc': np.float64,
        'short_debt': np.float64,
        'staff_sale': np.float64,
        'totdebt_invcap': np.float64
    }
    date_cols = ['public_date']
    df_fin_ratio_firms = pd.read_csv(filepath,
                                  # nrows=10000,
                                  parse_dates=date_cols,
                                  usecols=[*list(column_types.keys()), *date_cols],
                                  dtype=column_types,
                                  date_format='%Y-%m-%d')
    # Convert the percentage strings of column 'divyield' to float64
    df_fin_ratio_firms['divyield'] = df_fin_ratio_firms['divyield'].apply( lambda s: float( s[:-1] ) if s==s else np.nan )
    df_fin_ratio_firms['divyield'] = df_fin_ratio_firms['divyield'].astype(np.float64)
    # Convert public_date column to year-month column --> need for merging later
    df_fin_ratio_firms['year-month'] = df_fin_ratio_firms['public_date'].dt.to_period('M')
    df_fin_ratio_firms = df_fin_ratio_firms.drop( columns='public_date' )     # the 'public_date' column is no longer needed
    # Filter for relevant tickers
    df_fin_ratio_firms = df_fin_ratio_firms[ df_fin_ratio_firms['TICKER'].isin(relevant_tickers) ]
    # Rename columns
    cols_names = list(df_fin_ratio_firms.columns)
    cols_names = [ f'FinRatio::{col}' if (col!='TICKER' and col!='year-month') else col for col in cols_names]
    df_fin_ratio_firms.columns = cols_names
    # handle outliers
    for col in df_fin_ratio_firms.columns:
        if 'FinRatio' in col:
            df_fin_ratio_firms[col] = winsorize(df_fin_ratio_firms[col], limits=(0.01, 0.01))

    return df_fin_ratio_firms


# imports stock CSV and builds initial dataframe
# --> data type handling, pivoting, one-hot-encodings, log-returns calculation, dropping some columns, etc.
# parameter 'force_rebuild' can be used to ensure the database is rebuilt, even when parquet file already exists
def get_stock_data(force_rebuild=False):
    parquet_files = get_files_in_folder('export_dfs/', 'parquet')
    if 'export_dfs/stock_df.parquet' in parquet_files and not force_rebuild:
        return pd.read_parquet('export_dfs/stock_df.parquet')

    file_name = 'path_to_stock_data'    # provide csv
    column_types = {
        'TICKER': str,
        'PRC': np.float64,
        'VOL': np.float64,
        'BID': np.float64,
        'ASK': np.float64,
        'ASKHI': np.float64,
        'BIDLO': np.float64,
        'SHROUT': np.float64,
        'CFACPR': np.float64,
        'vwretd': np.float64,
        'vwretx': np.float64,
        'ewretd': np.float64,
        'ewretx': np.float64,
        'GICS Sector': str,
        'CAPEI_Mean': np.float64,
        'CAPEI_Median': np.float64,
        'GProf_Mean': np.float64,
        'GProf_Median': np.float64,
        'PEG_trailing_Mean': np.float64,
        'PEG_trailing_Median': np.float64,
        'accrual_Mean': np.float64,
        'accrual_Median': np.float64,
        'adv_sale_Mean': np.float64,
        'adv_sale_Median': np.float64,
        'aftret_eq_Mean': np.float64,
        'aftret_eq_Median': np.float64,
        'aftret_equity_Mean': np.float64,
        'aftret_equity_Median': np.float64,
        'aftret_invcapx_Mean': np.float64,
        'aftret_invcapx_Median': np.float64,
        'at_turn_Mean': np.float64,
        'at_turn_Median': np.float64,
        'bm_Mean': np.float64,
        'bm_Median': np.float64,
        'capital_ratio_Mean': np.float64,
        'capital_ratio_Median': np.float64,
        'cash_conversion_Mean': np.float64,
        'cash_conversion_Median': np.float64,
        'cash_debt_Mean': np.float64,
        'cash_debt_Median': np.float64,
        'cash_lt_Mean': np.float64,
        'cash_lt_Median': np.float64,
        'cash_ratio_Mean': np.float64,
        'cash_ratio_Median': np.float64,
        'cfm_Mean': np.float64,
        'cfm_Median': np.float64,
        'curr_debt_Mean': np.float64,
        'curr_debt_Median': np.float64,
        'curr_ratio_Mean': np.float64,
        'curr_ratio_Median': np.float64,
        'de_ratio_Mean': np.float64,
        'de_ratio_Median': np.float64,
        'debt_assets_Mean': np.float64,
        'debt_assets_Median': np.float64,
        'debt_at_Mean': np.float64,
        'debt_at_Median': np.float64,
        'debt_capital_Mean': np.float64,
        'debt_capital_Median': np.float64,
        'debt_ebitda_Mean': np.float64,
        'debt_ebitda_Median': np.float64,
        'debt_invcap_Mean': np.float64,
        'debt_invcap_Median': np.float64,
        'divyield_Mean': np.float64,
        'divyield_Median': np.float64,
        'dltt_be_Mean': np.float64,
        'dltt_be_Median': np.float64,
        'dpr_Mean': np.float64,
        'dpr_Median': np.float64,
        'efftax_Mean': np.float64,
        'efftax_Median': np.float64,
        'equity_invcap_Mean': np.float64,
        'equity_invcap_Median': np.float64,
        'evm_Mean': np.float64,
        'evm_Median': np.float64,
        'fcf_ocf_Mean': np.float64,
        'fcf_ocf_Median': np.float64,
        'gpm_Mean': np.float64,
        'gpm_Median': np.float64,
        'int_debt_Mean': np.float64,
        'int_debt_Median': np.float64,
        'int_totdebt_Mean': np.float64,
        'int_totdebt_Median': np.float64,
        'intcov_Mean': np.float64,
        'intcov_Median': np.float64,
        'intcov_ratio_Mean': np.float64,
        'intcov_ratio_Median': np.float64,
        'inv_turn_Mean': np.float64,
        'inv_turn_Median': np.float64,
        'lt_debt_Mean': np.float64,
        'lt_debt_Median': np.float64,
        'lt_ppent_Mean': np.float64,
        'lt_ppent_Median': np.float64,
        'npm_Mean': np.float64,
        'npm_Median': np.float64,
        'ocf_lct_Mean': np.float64,
        'ocf_lct_Median': np.float64,
        'opmad_Mean': np.float64,
        'opmad_Median': np.float64,
        'opmbd_Mean': np.float64,
        'opmbd_Median': np.float64,
        'pay_turn_Mean': np.float64,
        'pay_turn_Median': np.float64,
        'pcf_Mean': np.float64,
        'pcf_Median': np.float64,
        'pe_exi_Mean': np.float64,
        'pe_exi_Median': np.float64,
        'pe_inc_Mean': np.float64,
        'pe_inc_Median': np.float64,
        'pe_op_basic_Mean': np.float64,
        'pe_op_basic_Median': np.float64,
        'pe_op_dil_Mean': np.float64,
        'pe_op_dil_Median': np.float64,
        'pretret_earnat_Mean': np.float64,
        'pretret_earnat_Median': np.float64,
        'pretret_noa_Mean': np.float64,
        'pretret_noa_Median': np.float64,
        'profit_lct_Mean': np.float64,
        'profit_lct_Median': np.float64,
        'ps_Mean': np.float64,
        'ps_Median': np.float64,
        'ptb_Mean': np.float64,
        'ptb_Median': np.float64,
        'ptpm_Mean': np.float64,
        'ptpm_Median': np.float64,
        'quick_ratio_Mean': np.float64,
        'quick_ratio_Median': np.float64,
        'rd_sale_Mean': np.float64,
        'rd_sale_Median': np.float64,
        'rect_act_Mean': np.float64,
        'rect_act_Median': np.float64,
        'rect_turn_Mean': np.float64,
        'rect_turn_Median': np.float64,
        'roa_Mean': np.float64,
        'roa_Median': np.float64,
        'roce_Mean': np.float64,
        'roce_Median': np.float64,
        'roe_Mean': np.float64,
        'roe_Median': np.float64,
        'sale_equity_Mean': np.float64,
        'sale_equity_Median': np.float64,
        'sale_invcap_Mean': np.float64,
        'sale_invcap_Median': np.float64,
        'sale_nwc_Mean': np.float64,
        'sale_nwc_Median': np.float64,
        'short_debt_Mean': np.float64,
        'short_debt_Median': np.float64,
        'staff_sale_Mean': np.float64,
        'staff_sale_Median': np.float64,
        'totdebt_invcap_Mean': np.float64,
        'totdebt_invcap_Median': np.float64
    }
    date_cols = ['date']

    df_raw = pd.read_csv(file_name,
                     # nrows=10000,
                     parse_dates=date_cols,
                     usecols=[*list(column_types.keys()), *date_cols],
                     dtype=column_types,
                     date_format='%Y-%m-%d')
    df_raw = df_raw.drop_duplicates(subset=['TICKER', 'date'])

    # Get relevant tickers
    tickers = read_tickers("TICKERS/SP100_TICKER.txt")
    # Get tickers to exclude
    tickers_exclude = read_tickers("TICKERS/TICKER_exclude.txt")
    tickers_exclude = set([])
    # remove tickers with too little data
    tickers -= tickers_exclude

    # filter for SP100 companies and create year-month column
    df_raw = df_raw[df_raw['TICKER'].isin(tickers)]
    df_raw['year-month'] = df_raw['date'].dt.to_period('M')
    # obtain firm-level financial ratios
    df_fin_ratio_firms = get_finratio_firmlevel(relevant_tickers=tickers)
    # merge the two dfs ('df_raw' and 'df_fin_ratio_firms') on year-month and TICKER columns
    df = pd.merge( df_raw, df_fin_ratio_firms, how='left', on=['TICKER', 'year-month'] )

    # transform GICS sectors to one hot encodings
    df['GICS'] = df['GICS Sector'].copy().astype(str)  # keep copy of original column for later
    df = pd.get_dummies(df, columns=['GICS'])   # transform GICS sectors to one-hot encodings

    # apply pivoting
    df_pivot = df.pivot(index='date', columns='TICKER')

    # swap levels
    df_pivot = swap_levels(df_pivot)

    # add some simple features
    for ticker in df_pivot.columns.levels[0]:
        # calculate log-returns and while considering stock splittings
        prc_col = df_pivot.loc[:, (ticker, 'PRC')]
        cfacpr_col = df_pivot.loc[:, (ticker, 'CFACPR')]
        prc_mult = cfacpr_col.shift(1) / cfacpr_col
        LogRet = np.log(prc_col*prc_mult) - np.log(prc_col.shift(1))
        df_pivot.loc[:, (ticker, 'LogRet')] = LogRet
        # calculate bid-ask spreads
        ask_col, bid_col = df_pivot.loc[:, (ticker, 'ASK')], df_pivot.loc[:, (ticker, 'BID')]
        askhi_col, bidlo_col = df_pivot.loc[:, (ticker, 'ASKHI')], df_pivot.loc[:, (ticker, 'BIDLO')]
        df_pivot.loc[:, (ticker, 'ASK-BID')] = ask_col - bid_col
        df_pivot.loc[:, (ticker, 'ASKHI-BIDLO')] = askhi_col - bidlo_col

    # drop unnecessary columns:
    df_pivot = df_pivot.drop( columns=['PRC', 'ASK', 'BID', 'ASKHI', 'BIDLO', 'CFACPR'], level=1 )

    # save to disk
    df_pivot.to_parquet('export_dfs/stock_df.parquet')

    # return the dataframe
    return df_pivot
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == '__main__':
    tickers = read_tickers("TICKERS/SP100_TICKER.txt")
    get_finratio_firmlevel(tickers)

    # get stock data
    df = get_stock_data(force_rebuild=False)
    df = swap_levels(df)

    # get option data
    from Option_Features import get_option_data
    df_price, df_volsurf = get_option_data()

    df_volsurf = swap_levels(df_volsurf)

    df_volsurf = df_volsurf.drop(columns=['GOOG', 'RTX'], level=1)
    df_volsurf.columns = df_volsurf.columns.remove_unused_levels()

    # plot stock data availability
    plot_two_data_availabilities(df['LogRet'], df_volsurf['smile_volsurf::atm_ivol_custom'],
                                 'Stocks Data', 'Option Data', 'Data Availability',
                                 show=False, save=True, flip=True)
