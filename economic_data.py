import pandas as pd
import numpy as np
from Option_Price_Features import get_riskfree_rate
import os
import yfinance as yf


# GFD data
def get_macro_data():
    macro_filepath = 'path_to_macro_csv'        # provide file
    column_types = {
        'Nominal GDP': np.float64,
        'GDP 2017_dollars': np.float64,
        'Prime_rate': np.float64,
        'Economic_uncer': np.float64,
        'Equity_uncer': np.float64,
        'GFD_energy': np.float64,
        'GFD_argi': np.float64,
        'GFD_indus': np.float64,
        'Tin': np.float64,
        'Platinum': np.float64,
        'Copper': np.float64,
        'Aluminum': np.float64,
        'Nickle': np.float64,
        'CPI': np.float64,
        'Unemploy': np.float64,
        'WTI_Oil': np.float64,
    }  # we don't use all columns
    date_cols = ['Date']
    df_macro = pd.read_csv(macro_filepath,
                           parse_dates=date_cols,
                           usecols=[*list(column_types.keys()), *date_cols],
                           dtype=column_types,
                           date_format='%Y-%m-%d')
    # Forward fill the database with data
    df_macro = df_macro.fillna(method='ffill')
    # Filter out relevant time frame
    df_macro = df_macro[(df_macro['Date'] >= '2010-01-01') & (df_macro['Date'] <= '2019-12-31')]
    # Rename columns (prepend 'MACRO::' to every column except 'Date)
    df_macro.columns = [f'MACRO::{col}' if col!='Date' else col for col in df_macro.columns]
    return df_macro


# Exchange rates (from Federal Reserve database: https://www.federalreserve.gov/default.htm)
def get_FX_rates():
    fx_filepath = ('FRB_H10.csv')   # provide file
    # We include the following FX rates (most traded and influential economies in relation to the US):
    #   1.  EUR/USD (Euro)
    # 	2.	JPY/USD (Japanese Yen)
    # 	3.	GBP/USD (British Pound)
    # 	4.	CHF/USD (Swiss Franc)
    # 	5.	CAD/USD (Canadian Dollar)
    # EUR/USD and GBP/USD are not given directly but only as reciprocals ( USD/EUR and USD/GBP )
    exchange_cols = ['RXI$US_N.B.EU', 'RXI_N.B.JA', 'RXI$US_N.B.UK', 'RXI_N.B.SZ', 'RXI_N.B.CA']
    df_fx = pd.read_csv(fx_filepath,
                        parse_dates=['Time Period'],
                        usecols=[*['Time Period'], *exchange_cols],
                        date_format='%Y-%m-%d',
                        skiprows=5, header=0)
    df_fx[exchange_cols] = df_fx[exchange_cols].apply(pd.to_numeric, errors='coerce')
    # take reciprocal of FX rates for EUR and for GBP --> all FX rates per USD
    df_fx['RXI$US_N.B.EU'] = 1 / df_fx['RXI$US_N.B.EU']
    df_fx['RXI$US_N.B.UK'] = 1 / df_fx['RXI$US_N.B.UK']
    # rename the columns for more readability
    df_fx = df_fx.rename(columns={'Time Period': 'Date',
                                    'RXI$US_N.B.EU': 'FX::EUR/USD',
                                    'RXI_N.B.JA': 'FX::JPY/USD',
                                    'RXI$US_N.B.UK': 'FX::GBP/USD',
                                    'RXI_N.B.SZ': 'FX::CHF/USD',
                                    'RXI_N.B.CA': 'FX::CAD/USD'})
    return df_fx


# Load and merge EPU data (categorical and regular)
def get_EPU_data():
    path_EPU_categorical = ('Categorical_EPU_Data.xlsx')    # provide file
    path_EPU = ('US_Policy_Uncertainty_Data.xlsx')      # provide file
    # Load data
    df_EPU_categorical = pd.read_excel(path_EPU_categorical, header=0, skipfooter=1)
    df_EPU = pd.read_excel(path_EPU, header=0, skipfooter=1)
    # Add Date column for 'df_EPU'
    df_EPU['Date'] = pd.to_datetime(df_EPU[['Month', 'Year']].assign(DAY=1))
    df_EPU = df_EPU.drop( columns=['Month', 'Year'] )
    # Filter out relevant time frame
    df_EPU = df_EPU[ (df_EPU.Date >= '2010-01-01') & (df_EPU.Date <= '2019-12-31') ]
    df_EPU_categorical = df_EPU_categorical[ (df_EPU_categorical.Date >= '2010-01-01') &
                                             (df_EPU_categorical.Date <= '2019-12-31') ]
    # Add 'year-month' column for both dataframes and delete 'Date' column
    df_EPU_categorical['year-month'] = df_EPU_categorical['Date'].dt.to_period('M')
    df_EPU_categorical = df_EPU_categorical.drop(columns='Date')
    df_EPU['year-month'] = df_EPU['Date'].dt.to_period('M')
    df_EPU = df_EPU.drop(columns='Date')

    # Change column names for df_EPU
    cols_EPU = list(df_EPU.columns)
    cols_EPU = [f"EPU::{col}"
                if col!='year-month'
                else col
                for col in cols_EPU]
    df_EPU.columns = cols_EPU

    # Change column names for df_EPU_categorical
    cols_EPU_categorical = list(df_EPU_categorical.columns)
    cols_EPU_categorical = [f"EPU_Categorical::{(col.split('.')[-1])[1:]}"
                            if col!='year-month'
                            else col
                            for col in cols_EPU_categorical ]
    df_EPU_categorical.columns = cols_EPU_categorical

    # Merge the two dataframes ('df_EPU', 'df_EPU_categorical') on 'year-month'
    df_EPU_both = pd.merge( df_EPU, df_EPU_categorical, how='left', on='year-month' )
    return df_EPU_both


# Load VIX and VXO data
def get_vix_vxo():
    # load and format VIX data (https://fred.stlouisfed.org/series/VIXCLS)
    path_vix = ("VIXCLS.csv")       # provide file
    df_vix = pd.read_csv(path_vix, parse_dates=['DATE'], usecols=['DATE', 'VIXCLS'], date_format='%Y-%m-%d')
    df_vix['VIXCLS'] = df_vix['VIXCLS'].apply(pd.to_numeric, errors='coerce')
    df_vix.columns = ['Date', 'INDEX::VIX']
    # load and format VXO data (https://fred.stlouisfed.org/series/VXOCLS)
    path_vxo = ("VXOCLS.csv")       # provide file
    df_vxo = pd.read_csv(path_vxo, parse_dates=['DATE'], usecols=['DATE', 'VXOCLS'], date_format='%Y-%m-%d')
    df_vxo['VXOCLS'] = df_vxo['VXOCLS'].apply(pd.to_numeric, errors='coerce')
    df_vxo.columns = ['Date', 'INDEX::VXO']
    # merge both dataframes and return
    df_vix_vxo = pd.merge(df_vix, df_vxo, how='outer', on='Date')
    return df_vix_vxo


def get_SP100_data():
    # Download SP100 Index Price and Volume (from YAHOO Finance)
    df_sp100 = yf.download("^OEX", start="2010-01-01", end="2019-12-31")
    df_sp100 = df_sp100[['Close', 'Volume']]
    df_sp100.columns = ['INDEX::SP100_PRC', 'INDEX::SP100_Volume']
    df_sp100['Date'] = df_sp100.index
    df_sp100 = df_sp100.reset_index(drop=True)
    return df_sp100


def get_economic_data():
    # 1) import some general data (from Varin)
    df = get_macro_data()

    # 2) Collect EPU data and merge it with 'df'
    df_EPU = get_EPU_data()
    df['year-month'] = df['Date'].dt.to_period('M')
    df = pd.merge( df, df_EPU, how='left', on='year-month' )
    df = df.drop( columns='year-month' )

    # 3) Collect risk free rates (linearly interpolated data from OptionMetrics' Zero-Curve)
    zc_filepath = 'additional_dfs/zero_curve.parquet'  # zero-curve file from OptionMetrics (converted to parquet)
    assert os.path.isfile(zc_filepath)  # provide file
    zc_df = pd.read_parquet(zc_filepath)
    df['ZCB::rate_7_days'] = df.Date.apply(lambda d: get_riskfree_rate(d, 7, zc_df))  # one week
    df['ZCB::rate_30_days'] = df.Date.apply(lambda d: get_riskfree_rate(d, 30, zc_df))  # one month
    df['ZCB::rate_365_days'] = df.Date.apply(lambda d: get_riskfree_rate(d, 365, zc_df))  # one year
    df['ZCB::rate_3650_days'] = df.Date.apply(lambda d: get_riskfree_rate(d, 3650, zc_df))  # one decade

    # 4) Collect foreign exchange (FX) rates and merge with 'df'
    df_fx = get_FX_rates()
    df = pd.merge(df, df_fx, how='left', on='Date')

    # 5) Collect VIX and VXO data and merge it with 'df'
    df_vix_vxo = get_vix_vxo()
    df = pd.merge( df, df_vix_vxo, how='left', on='Date' )

    # 6) Collect S&P100 Index (OEX) data (closing price and volume) and merge it with 'df'
    df_sp100 = get_SP100_data()
    df = pd.merge( df, df_sp100, how='left', on='Date' )

    # 7) Append 'ECO::' to all columns except 'year-month', and return
    df = df.reset_index(drop=True)
    df.columns = ['ECO::' + col if col!='Date' else col for col in df.columns]
    return df


########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == '__main__':
    df_economic = get_economic_data()
