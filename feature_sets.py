
########################################################################################################################
# SET 1
# Base features (FinRatio, CRSP, ECO, EVENT, GICS)
########################################################################################################################
set1_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               ]
set1_include, set1_exclude = None, None


########################################################################################################################
# SET 2
# Base features + HIST data
########################################################################################################################
set2_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               'HIST::',        # estimators from GARCH, HAR, and weighted averages
               ]
set2_include, set2_exclude = None, None

########################################################################################################################
# SET 3
# Base features + Basic Option Data + Black-Scholes IVOLs
########################################################################################################################
set3_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               'OPTION_PRC::GEN::',     # general features
               'OPTION_VOLSURF::GEN::',     # general features
               ]
set3_include = None
set3_exclude = None

########################################################################################################################
# SET 4
# Base features + Basic Option Data + Model-free IVOLs
########################################################################################################################
set4_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               'OPTION_PRC::GEN::',     # general features
               'OPTION_PRC::MF::',      # model-free features
               'OPTION_VOLSURF::GEN::',     # general features
               'OPTION_VOLSURF::MF::',      # model-free features
               ]
set4_include = None
set4_exclude = [ 'OPTION_PRC::GEN::atm_ivol_custom',
                 'OPTION_PRC::GEN::atm_ivol_spline',
                 'OPTION_PRC::GEN::atm_slope_custom',
                 'OPTION_PRC::GEN::atm_slope_spline',
                 'OPTION_VOLSURF::GEN::atm_ivol_custom',
                 'OPTION_VOLSURF::GEN::atm_ivol_spline',
                 'OPTION_VOLSURF::GEN::atm_slope_custom',
                 'OPTION_VOLSURF::GEN::atm_slope_spline',
                 # only use 'trapz' version of model-free calculation
                 'OPTION_PRC::MF::volatility',
                 'OPTION_PRC::MF::skew',
                 'OPTION_PRC::MF::kurtosis',
                 'OPTION_VOLSURF::MF::volatility',
                 'OPTION_VOLSURF::MF::skew',
                 'OPTION_VOLSURF::MF::kurtosis'
                 ]

########################################################################################################################
# SET 5
# Base features + Basic Option Data (no IVOL) + custom smile parametrisation (Option_Prices file & Vol_Surfaces file)
########################################################################################################################
set5_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               'OPTION_PRC::GEN::',     # general features
               'OPTION_VOLSURF::GEN::',     # general features
               # price file curves
               'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::',     # CUSTOM: -Log(Moneyness) --> Impl
               'OPTION_PRC::CUSTOM_Moneyness_Impl::',       # CUSTOM: Moneyness --> Impl
               # volsurf file curves
               'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::',     # CUSTOM: -Log(Moneyness) --> Impl
               'OPTION_VOLSURF::CUSTOM_Moneyness_Impl::',    # CUSTOM: Moneyness --> Impl
               ]
set5_include = None
set5_exclude = [ 'OPTION_PRC::GEN::atm_ivol_custom',
                 'OPTION_PRC::GEN::atm_ivol_spline',
                 'OPTION_PRC::GEN::atm_slope_custom',
                 'OPTION_PRC::GEN::atm_slope_spline',
                 'OPTION_VOLSURF::GEN::atm_ivol_custom',
                 'OPTION_VOLSURF::GEN::atm_ivol_spline',
                 'OPTION_VOLSURF::GEN::atm_slope_custom',
                 'OPTION_VOLSURF::GEN::atm_slope_spline',
                 # remove implicit ATM IVOL information (in centre_y variable)
                 'OPTION_VOLSURF::CUSTOM_Moneyness_Impl::centre_y',
                 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::centre_y',
                 'OPTION_PRC::CUSTOM_Moneyness_Impl::centre_y',
                 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::centre_y',
                 # remove redundancy --> only include '::x_min', '::x_max' for Moneyness, not -LogMoneyness
                 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::x_max',
                 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::x_min',
                 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::x_max',
                 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::x_min'
                 ]

########################################################################################################################
# SET 6
# Base features + Basic Option Data (no IVOL) + spline smile parametrisation (Option_Prices file & Vol_Surfaces file)
########################################################################################################################
set6_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               'OPTION_PRC::GEN::',     # general features
               'OPTION_VOLSURF::GEN::',     # general features
               # price file curves
               'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::',     # SPLINE: -Log(Moneyness) --> Impl
               'OPTION_PRC::SPLINE_Moneyness_Impl::',       # SPLINE: Moneyness --> Impl
               # volsurf file curves
               'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::',     # SPLINE: -Log(Moneyness) --> Impl
               'OPTION_VOLSURF::SPLINE_Moneyness_Impl::'    # SPLINE: Moneyness --> Impl
               ]
set6_include = None
set6_exclude = [ 'OPTION_PRC::GEN::atm_ivol_custom',
                 'OPTION_PRC::GEN::atm_ivol_spline',
                 'OPTION_PRC::GEN::atm_slope_custom',
                 'OPTION_PRC::GEN::atm_slope_spline',
                 'OPTION_VOLSURF::GEN::atm_ivol_custom',
                 'OPTION_VOLSURF::GEN::atm_ivol_spline',
                 'OPTION_VOLSURF::GEN::atm_slope_custom',
                 'OPTION_VOLSURF::GEN::atm_slope_spline',
                 # remove redundancy --> only include '::x_min', '::x_max' for Moneyness, not -LogMoneyness
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::x_max',
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::x_min',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::x_max',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::x_min'
                 ]

########################################################################################################################
# SET 7
# Base features + Basic Option Data (no IVOL) + price smirk parametrisation (Option_Prices file & Vol_Surfaces file)
########################################################################################################################
set7_groups = ['CRSP::', 'ECO::', 'EVENT::', 'FinRatio::',
               'GICS ',         # use GICS Sector (column of strings --> categories) instead of One-Hot-Encodings
               'OPTION_PRC::GEN::',     # general features
               'OPTION_VOLSURF::GEN::',     # general features
               # price file curves
               'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::',     # SPLINE: -Log(Moneyness) --> Price
               'OPTION_PRC::SPLINE_Moneyness_Prc::',       # SPLINE: Moneyness --> Price
               # volsurf file curves
               'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::',     # SPLINE: -Log(Moneyness) --> Price
               'OPTION_VOLSURF::SPLINE_Moneyness_Prc::'    # SPLINE: Moneyness --> Price
               ]
set7_include = None
set7_exclude = [ 'OPTION_PRC::GEN::atm_ivol_custom',
                 'OPTION_PRC::GEN::atm_ivol_spline',
                 'OPTION_PRC::GEN::atm_slope_custom',
                 'OPTION_PRC::GEN::atm_slope_spline',
                 'OPTION_VOLSURF::GEN::atm_ivol_custom',
                 'OPTION_VOLSURF::GEN::atm_ivol_spline',
                 'OPTION_VOLSURF::GEN::atm_slope_custom',
                 'OPTION_VOLSURF::GEN::atm_slope_spline',
                 # SPLINE: ... --> Prc
                 # remove redundancy --> only include '::x_min', '::x_max' for Moneyness, not -LogMoneyness
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::x_max',
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::x_min',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::x_max',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::x_min'
                 ]

########################################################################################################################
# SET 8
# All
########################################################################################################################
set8_groups = 'ALL'
set8_include = None
set8_exclude = [ # SPLINE: ... --> Prc
                 # remove redundancy --> only include '::x_min', '::x_max' for Moneyness, not -LogMoneyness
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::x_max',
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Prc::x_min',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::x_max',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Prc::x_min',
                 # SPLINE: ... --> Impl
                 # remove redundancy --> only include '::x_min', '::x_max' for Moneyness, not -LogMoneyness
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::x_max',
                 'OPTION_VOLSURF::SPLINE_-Log(Moneyness)_Impl::x_min',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::x_max',
                 'OPTION_PRC::SPLINE_-Log(Moneyness)_Impl::x_min',
                 # CUSTOM: ... --> Impl
                 # remove redundancy --> only include '::x_min', '::x_max' for Moneyness, not -LogMoneyness
                 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::x_max',
                 'OPTION_VOLSURF::CUSTOM_-Log(Moneyness)_Impl::x_min',
                 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::x_max',
                 'OPTION_PRC::CUSTOM_-Log(Moneyness)_Impl::x_min',
                 # MODEL-FREE stuff
                 # only use 'trapz' version of model-free calculation
                 'OPTION_PRC::MF::volatility',
                 'OPTION_PRC::MF::skew',
                 'OPTION_PRC::MF::kurtosis',
                 'OPTION_VOLSURF::MF::volatility',
                 'OPTION_VOLSURF::MF::skew',
                 'OPTION_VOLSURF::MF::kurtosis'
                 ]


########################################################################################################################
# Functions
########################################################################################################################
# extract the different categories of the df (given by the '::' structure in the column names)
def get_categories_columns(df, sub_categories=True):
    columns = list(df.columns)
    if sub_categories:
        columns = [ col.split('::')[:-1] + [''] if '::' in col else [col] for col in columns]
        columns = ['::'.join(col) for col in columns]
    else:
        columns = [col.split('::')[0] if '::' in col else col for col in columns]
    columns = set(columns)
    columns -= {'ticker', 'date', 'one_month_lookahead_vol'}
    return columns


def get_columns_in_category(df, category):
    columns = list(df.columns)
    cols_in_cat = [col for col in columns if category in col]
    return cols_in_cat


# create set of column names (given list of groups to include, and specific columns to include/exclude)
def make_set(df, groups, include_columns=None, exclude_columns=None):
    if include_columns is None:
        include_columns = []
    if exclude_columns is None:
        exclude_columns = []
    columns_all = list(df.columns)

    if type(groups)==str and groups == 'ALL':
        columns_groups = columns_all
    else:
        assert type(groups) == list
        # collect all columns to include from the groups argument
        columns_groups = []
        for group_prefix in groups:
            cols = [col for col in columns_all if group_prefix in col]
            columns_groups = [*columns_groups, *cols]

    # include/ exclude specific columns
    columns_selection = ( set(columns_groups) | set(include_columns) ) - set(exclude_columns)
    # add standard columns ('date', 'ticker', 'one_month_lookahead_vol'), sort, and return
    columns_selection = columns_selection - {'date', 'ticker', 'one_month_lookahead_vol'}
    columns_selection = ['date', 'ticker', 'one_month_lookahead_vol'] + sorted( list(columns_selection) )
    return columns_selection


# get set from definitions in this file (see above)
def get_set(df, n):
    set_dict = {
        1: [set1_groups, set1_include, set1_exclude],
        2: [set2_groups, set2_include, set2_exclude],
        3: [set3_groups, set3_include, set3_exclude],
        4: [set4_groups, set4_include, set4_exclude],
        5: [set5_groups, set5_include, set5_exclude],
        6: [set6_groups, set6_include, set6_exclude],
        7: [set7_groups, set7_include, set7_exclude],
        8: [set8_groups, set8_include, set8_exclude]
    }
    set_groups, set_include, set_exclude = set_dict[n]
    return make_set(df, set_groups, include_columns=set_include, exclude_columns=set_exclude)


if __name__=='__main__':
    # load feature database
    from merge_databases import get_df_merged
    df_merged = get_df_merged(force_rebuild=False)
    # display all features available
    for col in df_merged.columns:
        print(col)
    print(f'Total number of columns: {len(df_merged.columns)}')
    print('##############################')
    print('##############################')
    print('##############################')
    # display features of the different sets
    for n in range(8, 9):
        set_n = get_set(df_merged, n)
        for col in set_n:
            print(col)
        print(f'Number of columns in Set {n}: {len(set_n)}')
        print('##############################')
        print('##############################')
        print('##############################')

    print('##############################')
    print('##############################')
    print('##############################')
    print('##############################')
    print('##############################')
    print('##############################')

    cat_cols = get_columns_in_category(df_merged, 'EVENT::')
    print(cat_cols)
    print(f'Number of columns in Category: {len(cat_cols)}')
