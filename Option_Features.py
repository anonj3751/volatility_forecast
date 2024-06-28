import pandas as pd
from helper import get_files_in_folder
from helper import unpivot
from helper import swap_levels
from plotting import plot_data_availability


def get_option_data():
    path_option_price_features = 'export_dfs/Option_Prices_Results'  # path to export from Option_Price_Features file
    path_option_volsurf_features = 'export_dfs/Option_VolSurface_Results'  # path to export from Option_VolSurf_Features file

    option_price_files = get_files_in_folder(path_option_price_features, 'parquet')
    option_volsurf_files = get_files_in_folder(path_option_volsurf_features, 'parquet')

    dfs_price = [ unpivot(pd.read_parquet(file)) for file in option_price_files ]
    dfs_volsurf = [ unpivot(pd.read_parquet(file)) for file in option_volsurf_files ]

    df_price = pd.concat(dfs_price, axis='rows')
    df_volsurf = pd.concat(dfs_volsurf, axis='rows')

    df_price = df_price.pivot(columns='ticker', index='date')
    df_price = swap_levels(df_price)
    df_volsurf = df_volsurf.pivot(columns='ticker', index='date')
    df_volsurf = swap_levels(df_volsurf)

    df_price = df_price.sort_index()
    df_volsurf = df_volsurf.sort_index()

    # df_price, df_volsurf = unpivot(df_price), unpivot(df_volsurf)

    return df_price, df_volsurf


if __name__ == '__main__':
    df_price, df_volsurf = get_option_data()
    df_price = swap_levels(df_price)
    df_price = df_price[ df_price.index < '2019-12-31' ]
    plot_data_availability(df_price['smile_0::time_to_mat'], 'Option Price Data Availability', show=True, save=False)
    df_volsurf = swap_levels(df_volsurf)
    plot_data_availability(df_volsurf['smile_volsurf::atm_ivol_custom'], 'Option Volatility-Surface Data Availability', show=True, save=False)