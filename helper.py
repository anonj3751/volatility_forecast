import os
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np


def unpivot(df):
    return df.stack(level=0).rename_axis(['date','ticker']).reset_index()


def swap_levels(df):
    # swap levels
    df.columns = df.columns.swaplevel(0, 1)
    return df.sort_index(axis=1, level=0)


def get_files_in_folder(dir, extension):
    assert len(dir) > 0
    if dir[-1] != '/':
        dir = dir + '/'
    files = os.listdir(dir)
    files = [dir + f for f in files if f.split('.')[-1]==extension]
    return files


########################################################################################################################
# CUSTOM FUNCTION
########################################################################################################################

# smoothstep function
def smooth_sign(x_in, alpha=0.1):
    x = (x_in + 0.5 * alpha) / alpha
    return np.where(x <= 0, 0, np.where(x >= 1, 1, 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3))


# parametric function controlling:
# - slope & curvature for the left and right part,
# - transition smoothness between two parts,
# - translations in x and y direction
def custom_function(x, a, b, c, d, x0, y0, alpha):
    y_neg = a * (x - x0) ** 2 + b * (x - x0)
    y_pos = c * (x - x0) ** 2 + d * (x - x0)
    # combining both parts
    y_right = np.where((x - x0) < -alpha, 0, y_pos)
    y_left = np.where((x - x0) > alpha, 0, y_neg)
    y = (1 - smooth_sign((x - x0), alpha)) * y_left + smooth_sign((x - x0), alpha) * y_right
    return y + y0


########################################################################################################################
#   STATISTICAL TESTS
########################################################################################################################

# Stationarity test (ADFuller)
# see https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
def compute_adfuller(data, print_details=True):
    result = adfuller(data.values)
    if print_details:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
            print("\u001b[32mStationary\u001b[0m")
        else:
            print("\x1b[31mNon-stationary\x1b[0m")
    return result[1]  # p-value

