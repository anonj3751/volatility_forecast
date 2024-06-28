import os
import pandas as pd
from plotting import plot_data_availability
from helper import swap_levels
from helper import get_files_in_folder

def process_factset_data(fact_set_base):
    file_names = os.listdir(fact_set_base)

    # Read all EXCEL files and convert to single DataFrame
    df_arr = []
    for file_name in file_names:
        if len(file_name) > 4 and file_name[-4:] == '.xls':
            file = pd.read_excel(fact_set_base + file_name, skiprows=3)
            df_arr.append(file)
    df = pd.concat(df_arr)

    # remove duplicate and empty rows
    # when merger happened, database sometimes includes the same event for different company names --> use 'subset=...'
    df = df.drop_duplicates(subset=['Date', 'Ticker', 'Event Type'])
    df = df.dropna(subset=['Date', 'Ticker'])

    # remove '-US' from ticker to match ticker format of other database
    df['Ticker'] = df['Ticker'].apply(lambda s: s[:-3])

    tickers_factset = list(set(df['Ticker']))
    tickers_factset.sort()

    # only keep relevant columns
    df = df[['Date', 'Ticker', 'Company Name', 'Event Type']]

    # change indexing for easier use later
    df = df.sort_values('Date')
    df = df.set_index(['Ticker', 'Date'])

    # should be 15144 entries (displayed in FactSet database for 1 January 2010 - 18 April 2024)
    print(f"Number of events in database: {len(df)}")

    # print all instances where company did not have four earning calls a year
    for ticker in tickers_factset:
        print(ticker)
        print("Not 4 Earning Releases in the following years:")
        my_df = df.loc[ticker].copy()
        my_df['Date'] = my_df.index
        my_df['Year'] = my_df['Date'].apply(lambda d: d.year)
        for year in range(2010, 2025):
            num_earning_releases = len(my_df[(my_df['Year'] == year) & (my_df['Event Type'] == 'Earnings Release')])
            if not num_earning_releases == 4:
                print(f"Year {year} --> Earning Releases: {num_earning_releases}")
        print('----------------------------------------------------------------------------')

    # save dataframe to disk
    df.to_parquet('export_dfs/earning_dates.parquet')


def get_earning_data(force_rebuild=False):
    parquet_files = get_files_in_folder('export_dfs/', 'parquet')
    if 'export_dfs/earning_dates.parquet' in parquet_files and not force_rebuild:
        df = pd.read_parquet('export_dfs/earning_dates.parquet')
    else:
        fact_set_base = 'base_path_to_factset'      # provide path
        process_factset_data(fact_set_base=fact_set_base)

    # prepare dataframe and perform pivoting
    tickers = df.index.levels[0]
    df_arr = []
    for ticker in tickers:
        df.loc[ticker, 'ticker'] = ticker
        df_arr.append(df.loc[ticker])
    df_all = pd.concat(df_arr, axis='rows')
    df_all = df_all.reset_index(drop=False)
    df_all = df_all.drop('Company Name', axis='columns')
    df_all = df_all.drop('Event Type', axis='columns')
    df_all = df_all.drop_duplicates(ignore_index=True)

    df_all['event_happened'] = df_all['ticker'].notna().astype(int)
    df_all_pivot = df_all.pivot(index='Date', columns='ticker')

    # swap levels
    df_all_pivot = swap_levels(df_all_pivot)

    return df_all_pivot


if __name__=='__main__':
    # plot data availability
    df_all_pivot = get_earning_data(force_rebuild=False)
    df_all_pivot = swap_levels(df_all_pivot)
    plot_data_availability(df_all_pivot['event_happened'], 'Earning Date Availability', show=True, save=False)
