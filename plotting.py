import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections.abc import Iterable
from sklearn import linear_model
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# helper functions for data availability plotting --> see stackoverflow
class AxTransformer:
    def __init__(self, datetime_vals=False):
        self.datetime_vals = datetime_vals
        self.lr = linear_model.LinearRegression()

        return

    def process_tick_vals(self, tick_vals):
        if not isinstance(tick_vals, Iterable) or isinstance(tick_vals, str):
            tick_vals = [tick_vals]

        if self.datetime_vals == True:
            tick_vals = pd.to_datetime(tick_vals).astype(int).values

        tick_vals = np.array(tick_vals)

        return tick_vals

    def fit(self, ax, axis='x'):
        axis = getattr(ax, f'get_{axis}axis')()

        tick_locs = axis.get_ticklocs()
        tick_vals = self.process_tick_vals([label._text for label in axis.get_ticklabels()])

        self.lr.fit(tick_vals.reshape(-1, 1), tick_locs)

        return

    def transform(self, tick_vals):
        tick_vals = self.process_tick_vals(tick_vals)
        tick_locs = self.lr.predict(np.array(tick_vals).reshape(-1, 1))

        return tick_locs


def set_date_ticks(ax, start_date, end_date, axis='x', date_format='%Y-%m-%d', **date_range_kwargs):
    dt_rng = pd.date_range(start_date, end_date, **date_range_kwargs)

    ax_transformer = AxTransformer(datetime_vals=True)
    ax_transformer.fit(ax, axis=axis)

    getattr(ax, f'set_{axis}ticks')(ax_transformer.transform(dt_rng))
    getattr(ax, f'set_{axis}ticklabels')(dt_rng.strftime(date_format))

    ax.tick_params(axis=axis, which='both', bottom=True, top=False, labelbottom=True)
    plt.xticks(rotation=45)

    return ax

######


# plot heatmap of available data (x: dates, y: tickers)
def plot_data_availability(series, title, show=False, save=True):
    df_presence = series.notna().astype(int)

    fig = plt.figure(figsize=(30, 20))
    cmap = mcolors.ListedColormap(['#c7c7c7', '#024f2f'])  # colours for 0 and 1
    ax = sns.heatmap(df_presence.T, cmap=cmap, cbar_kws={'ticks': [0, 1]})

    # Ensure the date ticks are set properly
    start_date = series.index.min()
    end_date = series.index.max()
    set_date_ticks(ax, start_date, end_date, freq='3MS')

    plt.title(f"{title}", fontsize=40, pad=40)
    plt.xlabel('Date', fontsize=25, labelpad=20)
    plt.ylabel('Ticker', fontsize=25, labelpad=20)
    # Adding horizontal grid lines
    ax.yaxis.grid(True, which='major', color='#c7c7c7', linestyle='dashed', linewidth=2, alpha=0.4)

    # Adjusting the colorbar to show binary values without gradient
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.set_yticklabels(['No Data', 'Data'])
    colorbar.ax.set_ylabel('Data Availability', fontsize=20, labelpad=40)

    plt.tight_layout()

    if save:
        plt.savefig(f'Images/{title}.png', dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_two_data_availabilities(series1, series2, title1, title2, title, show=False, save=True, flip=False):
    df_presence1 = series1.notna().astype(int)
    df_presence2 = series2.notna().astype(int)

    fig, axs = plt.subplots(1, 3, figsize=(50, 20),
                            gridspec_kw={'width_ratios': [1, 1, 0.15]},
                            constrained_layout=True)
    cmap = mcolors.ListedColormap(['#c7c7c7', '#024f2f'])  # colours for 0 and 1

    # create heatmaps
    heatmap_1 = sns.heatmap(df_presence1.T, cmap=cmap, cbar=False, ax=axs[0])
    heatmap_2 = sns.heatmap(df_presence2.T, cmap=cmap, cbar=False, ax=axs[1])
    # Plot for series1
    ax1 = heatmap_1
    start_date1 = series1.index.min()
    end_date1 = series1.index.max()
    set_date_ticks(ax1, start_date1, end_date1, freq='3MS', axis='x')
    axs[0].set_title(f"{title1}", fontsize=30, pad=20)
    axs[0].set_xlabel('Date', fontsize=25, labelpad=20)
    axs[0].set_ylabel('Ticker', fontsize=25, labelpad=20)
    axs[0].yaxis.grid(True, which='major', color='#c7c7c7', linestyle='dashed', linewidth=2, alpha=0.4)

    # Plot for series2
    ax2 = heatmap_2
    start_date2 = series2.index.min()
    end_date2 = series2.index.max()
    set_date_ticks(ax2, start_date2, end_date2, freq='3MS', axis='x')
    axs[1].set_title(f"{title2}", fontsize=30, pad=20)
    axs[1].set_xlabel('Date', fontsize=25, labelpad=20)
    axs[1].set_ylabel('', fontsize=25, labelpad=20)
    axs[1].yaxis.grid(True, which='major', color='#c7c7c7', linestyle='dashed', linewidth=2, alpha=0.4)

    # Create a colorbar axis and place it on the right
    cbar_ax = fig.add_axes([0.94, 0.105, 0.02, 1651/2000])
    colorbar = fig.colorbar(ax2.collections[0], cax=cbar_ax, ticks=[0, 1])
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.set_yticklabels(['No Data', 'Data'])
    colorbar.ax.set_ylabel('Data Availability', fontsize=30, labelpad=0)
    # Hide xaxis and yaxis, as well as border from right-most subplot
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['bottom'].set_visible(False)
    axs[2].spines['left'].set_visible(False)

    axs[0].set_xticks(axs[0].get_xticks(), rotation=45, labels=axs[0].get_xticklabels())
    axs[1].set_xticks(axs[1].get_xticks(), rotation=45, labels=axs[1].get_xticklabels())

    # add some space between subplots
    fig.subplots_adjust(wspace=0.4)
    plt.suptitle(title, fontsize=55, y=1.07)
    #plt.tight_layout()



    if save:
        plt.savefig(f'Images/{title}.png', dpi=150, bbox_inches='tight', pad_inches=3)
    if show:
        plt.show()
    plt.close(fig)


def plot_implied_vol_curves(df, ticker, date,
                            x_column, y_column,
                            x_fact = 1., y_fact = 1.,
                            call_put_symbols = True,
                            plot_regression = True, deg=5,
                            max_mat=365, num_mat=3, extrapolate_buffer=1):
    ticker_df = df[(df['ticker'] == ticker) & (df['date'] == date) & (df['time_to_mat_calendar'] < max_mat)]
    categories = ticker_df['time_to_mat_calendar'].unique()

    colors = plt.cm.jet(np.linspace(0, 1, len(categories)))
    plt.figure(figsize=(12, 6), dpi=200)
    mpl.style.use('seaborn-v0_8')

    # we only loop through the first >>num_mat<< categories (the >>num_mat<< smallest maturities).
    for i, category in enumerate(categories[:num_mat]):
        color = colors[i]
        subset = ticker_df[ticker_df['time_to_mat_calendar'] == category]
        if call_put_symbols:
            subset_call = subset[subset['cp_flag']=='C']
            plt.scatter(subset_call[x_column] * x_fact, subset_call[y_column] * y_fact, s=15, c=[color], alpha=0.8,
                        label=f'{category} days (Call)', marker='^')
            subset_put = subset[subset['cp_flag'] == 'P']
            plt.scatter(subset_put[x_column] * x_fact, subset_put[y_column] * y_fact, s=15, c=[color], alpha=0.8,
                        label=f'{category} days (Put)', marker='o')
        else:
            plt.scatter(subset[x_column] * x_fact, subset[y_column] * y_fact, s=10, c=[color], alpha=0.3)

        if plot_regression:
            model = np.poly1d(np.polyfit(subset[x_column]*x_fact, subset[y_column]*y_fact, deg=deg))
            min_x = np.min(subset[x_column]) * x_fact
            max_x = np.max(subset[x_column]) * x_fact
            polyline = np.linspace(min_x - extrapolate_buffer, max_x + extrapolate_buffer, 1000)
            plt.plot(polyline, model(polyline), c=color, label=f"{category} days")

    plt.legend(title='Time to maturity')
    plt.title('Implied Volatility Smile by Maturity')
    plt.suptitle(f'{ticker} on {date}', fontsize='xx-large')
    plt.xlabel('Strike price')
    plt.ylabel('Implied volatility')
    plt.show()


def plot_regressed_implied_vol_curves(df, ticker, date, axhlines=None, plot_stock_price = True, num_mat=3):
    if axhlines is None:
        axhlines = []
    colors = plt.cm.jet(np.linspace(0, 1, num_mat+1))
    plt.figure(figsize=(12, 6), dpi=200)
    mpl.style.use('seaborn-v0_8')

    stock_price = df.loc[date, ticker].PRC
    for j in range(1, num_mat+1):
        color = colors[j-1]
        curve_dict = df.loc[date, ticker][f'c{j}']
        maturity = int(curve_dict['maturity'])
        max_strike = curve_dict['max_strike']
        min_strike = curve_dict['min_strike']
        coefficients = curve_dict['coef']
        x_vals = np.linspace(min_strike, max_strike, 1000)
        y_vals = np.polyval(coefficients, x_vals)
        plt.plot(x_vals, y_vals, c=color, label=f"{maturity} days")
    if plot_stock_price:
        plt.axvline(stock_price, linewidth=3, linestyle=':', color=colors[num_mat], label='Spot price')
    if axhlines:
        for idx, y in enumerate(axhlines):
            plt.axhline(y, linewidth=1, linestyle=':', color=colors[idx])

    plt.legend(title='Time to maturity')
    plt.title('Regressed Implied Volatility Smile by Maturity')
    plt.suptitle(f'{ticker} on {date}', fontsize='xx-large')
    plt.xlabel('Strike price')
    plt.ylabel('Implied volatility')
    plt.show()


    plt.figure(figsize=(12, 6), dpi=200)
    mpl.style.use('seaborn-v0_8')
    plt.scatter(df.f1_train, df.f1_val)
    plt.title('Gridsearch Result')
    plt.xlabel('F1 Score Train')
    plt.ylabel('F1 Score Validation')
    plt.show()



def plot_regression_result(ticker, all_df, my_df,my_preds, dataset_label='Test Set', save_dir=None):
    idx_comp = ((my_df['ticker'] == ticker))
    df_comp = my_df[idx_comp]
    preds_comp = my_preds[idx_comp]
    df_comp_all = all_df[(all_df['ticker'] == ticker)]

    plt.figure(figsize=(12, 6), dpi=200)
    mpl.style.use('seaborn-v0_8')
    plt.plot(df_comp_all['date'], df_comp_all['target'],
             label='ground truth (30 day lookahead vol)', zorder=1)
    #plt.plot(df_comp_all['date'], df_comp_all['HIST::Volatility'],
    #         label='Volatility', zorder=1)
    plt.scatter(df_comp['date'], preds_comp, label='prediction (30 day lookahead vol)',
                color='red', marker='x', s=1, zorder=2)
    # plt.scatter(df_comp['date'], df_comp['Volatility'], label='Volatility')
    plt.title(f'XGBoost {dataset_label} ({ticker}) -- Regression Performance')
    plt.xlabel('Time')
    plt.ylabel('Volatility (in %)')
    plt.legend()
    if save_dir:
        plt.savefig(save_dir)
    plt.show()


def plot_binary_classification_result(ticker, all_df, my_df, my_preds, dataset_label='Test Set', save_dir=None):
    idx_comp = ((my_df['ticker'] == ticker))
    df_comp = my_df[idx_comp]
    preds_comp = my_preds[idx_comp]
    df_comp_all = all_df[(all_df['ticker'] == ticker)]

    plt.figure(figsize=(12, 6), dpi=200)
    mpl.style.use('seaborn-v0_8')
    plt.plot(df_comp_all['date'], df_comp_all['30_day_lookahead_vol']/100,
             label='Volatility (30 day lookahead)', zorder=1)
    plt.plot(df_comp_all['date'], df_comp_all['ATM_impl_vol_3_30_day_lookahead']*0.75,
             label='ATM implied volatility (30 day lookahead) * 1.1', zorder=1)

    plt.plot(df_comp_all['date'], df_comp_all['ATM_impl_vol_3_30_day_lookahead']*0.75 - df_comp_all['30_day_lookahead_vol']/100,
             label='Difference', zorder=1)

    #plt.plot(df_comp_all['date'], df_comp_all['ATM_impl_vol_3'],
    #         label='ATM implied volatility (3)', zorder=1)
    plt.plot(df_comp_all['date'], df_comp_all['target']*0.7,
             label='ground truth', zorder=1, alpha=0.4)

    # plot predictions
    pred_dates = my_df.iloc[my_preds.nonzero()]
    pred_dates = pred_dates[pred_dates['ticker'] == ticker]
    plt.scatter(pred_dates['date'], pred_dates['Volatility']/100, label='prediction (Vol in 30 days > ATM Vol (3))',
                color='black', marker='x', s=50, zorder=2)
    # plt.scatter(df_comp['date'], df_comp['Volatility'], label='Volatility')
    plt.title(f'XGBoost {dataset_label} ({ticker}) -- Binary Classification Performance')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    if save_dir:
        plt.savefig(save_dir)
    plt.show()


def plot_confusion_matrix(my_y, my_preds, dataset_label='Test Set', title='', save_dir=None, show=True):
    my_cm = confusion_matrix(my_y, my_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(my_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    if save_dir:
        plt.savefig(save_dir)
    if show:
        plt.show()


# Function to plot ACF and PACF and display or save to disk

def plot_ACF(data, num_of_lags, ticker='', show=True, basepath=''):
    # Autocorrelation plots
    plt.figure(figsize=(10, 5), dpi=300)
    plot_acf(data, lags=num_of_lags, ax=plt.gca())
    plt.title(f'{ticker} -- Autocorrelation Function')
    if show:
        plt.show()
    elif basepath != '':
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        plt.savefig(f'{basepath}/{ticker}_acf.png')


def plot_PACF(data, num_of_lags, ticker='', show=True, basepath=''):
    # Partial autocorrelation plots
    plt.figure(figsize=(10, 5), dpi=300)
    plot_pacf(data, lags=num_of_lags, ax=plt.gca())
    plt.title(f'{ticker} -- Partial Autocorrelation Function')
    if show:
        plt.show()
    elif basepath != '':
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        plt.savefig(f'{basepath}/{ticker}_pacf.png')
