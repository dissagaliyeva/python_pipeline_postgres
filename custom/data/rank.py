import pandas as pd
# from viz.visualize import plot_msno, plot_rank_matrix


def get_ranking(df, n=30, period='Q', markets=['ASIA FO (PVO)'], vol_plot_on=False, rank_plot_on=False):
    # filter data: by markets
    fltr = df.market.isin(markets)
    df = df[fltr]

    # set min period for rolling as 20 days to leave more training data
    min_n = 20
    # predefine name for pivot
    company = 'market_maker_mnemonic'
    quantity = 'order_quantity'
    date = 'order_date'

    '''
    Daily volume
    '''
    # pivot is key to align dates
    vol_pivot = df.pivot_table(index=date, columns=company, values=quantity, aggfunc='sum')

    # Sorting columns based on non-NA values count
    sorted_columns = vol_pivot.count().sort_values(ascending=False).index
    vol_pivot = vol_pivot[sorted_columns]

    # if vol_plot_on:
    #     print('>>>> vol_pivot')
    #     print(f'>>>> the company that has the highest non na columns:{list(pivot.columns)[:10]}')
    #     plot_msno(vol_pivot)  # 0 as nan

    # ranking frequency: use to resample code
    # resample to aggregate the volume by month

    '''
    Rolling sum volume
    '''
    # rolling sum
    # fill na for calc rolling
    vol_pivot = vol_pivot.fillna(0)  # <------ fill na here
    # calc rolling, set min period for rolling to e.g. 30
    vol_pivot_rolling = vol_pivot.rolling(window=n, min_periods=min_n).sum().dropna()  # <------dropna here

    if vol_plot_on:
        print('>>>> vol_pivot_rolling')
        plot_msno(vol_pivot_rolling)  # 0 as nan

    '''
    Resample
    '''
    # ranking frequency: use to resample code
    # resample to aggregate the volume
    # period = 'M'

    vol_pivot_rolling_resamp = vol_pivot_rolling.resample(period).mean()  # monthly mean of rolling n days
    
    if vol_plot_on:
        print('>>>> vol_pivot_rolling_resamp')
        plot_msno(vol_pivot_rolling_resamp)

    '''
    Create rank
    '''
    #
    vol_melt_rolling_resamp = vol_pivot_rolling_resamp.melt(ignore_index=False).reset_index()
    vol_melt_rolling_resamp['rank'] = vol_melt_rolling_resamp.groupby(date)['value'].rank(method='min', ascending=False)
    rank_raw = vol_melt_rolling_resamp.sort_values(by=['order_date', 'rank'], ascending=[True, True])
    
    if rank_plot_on:
        print('>>>> rank matrix')
        plt = plot_rank_matrix(rank_raw)
        plt.show
    return rank_raw


def transform_rank_fill_order_dates(last_order, rank, markets=None):
    '''
    this function
    '''

    # TODO
    # check ranks markets
    # warning if rank markets and markets does not match
    # pivot rank
    rank_pivot = rank.pivot_table(index='order_date', columns='market_maker_mnemonic', values='rank', aggfunc='mean')

    # create a dataframe with only the dates
    lo_dates = pd.DataFrame(index=last_order.order_date)

    # join all teh dates in last_order
    rank_merged = pd.merge(lo_dates, rank_pivot, how='left', left_index=True, right_index=True)

    # first forward fill data, then backward fill data
    # forward fill will use the latest resampled ranking from the dataset, e.g. end of previous month. This will prevent data leakage
    # backward fill will fill the missing date's value
    rank_merged = rank_merged.ffill().bfill()

    # rank long = [index, market maker, rank] there is no market here any more
    rank_filled = rank_merged.melt(ignore_index=False, var_name='market_maker_mnemonic', value_name='rank')

    return rank_filled


# import pandas as pd
# from custom.viz.visualize import plot_msno, plot_rank_matrix


# def get_ranking(df, n=30, period='Q', markets=['ASIA FO (PVO)'], vol_plot_on=False, rank_plot_on=False):
#     # filter data: by markets
#     fltr = df.market.isin(markets)
#     df = df[fltr]

#     # set min period for rolling as 20 days to leave more training data
#     min_n = 20
#     # predefine name for pivot
#     company = 'market_maker_mnemonic'
#     quantity = 'order_quantity'
#     date = 'order_date'

#     '''
#     Daily volume
#     '''
#     # pivot is key to align dates
#     vol_pivot = df.pivot_table(index=date, columns=company, values=quantity, aggfunc='sum')

#     # Sorting columns based on non-NA values count
#     sorted_columns = vol_pivot.count().sort_values(ascending=False).index
#     vol_pivot = vol_pivot[sorted_columns]

#     # if vol_plot_on:
#     #     print('>>>> vol_pivot')
#     #     print(f'>>>> the company that has the highest non na columns:{list(pivot.columns)[:10]}')
#     #     plot_msno(vol_pivot)  # 0 as nan

#     # ranking frequency: use to resample code
#     # resample to aggregate the volume by month

#     '''
#     Rolling sum volume
#     '''
#     # rolling sum
#     # fill na for calc rolling
#     vol_pivot = vol_pivot.fillna(0)  # <------ fill na here
#     # calc rolling, set min period for rolling to e.g. 30
#     vol_pivot_rolling = vol_pivot.rolling(window=n, min_periods=min_n).sum().dropna()  # <------dropna here

#     if vol_plot_on:
#         print('>>>> vol_pivot_rolling')
#         plot_msno(vol_pivot_rolling)  # 0 as nan

#     '''
#     Resample
#     '''
#     # ranking frequency: use to resample code
#     # resample to aggregate the volume
#     # period = 'M'

#     vol_pivot_rolling_resamp = vol_pivot_rolling.resample(period).mean()  # monthly mean of rolling n days
    
#     if vol_plot_on:
#         print('>>>> vol_pivot_rolling_resamp')
#         plot_msno(vol_pivot_rolling_resamp)

#     '''
#     Create rank
#     '''
#     #
#     vol_melt_rolling_resamp = vol_pivot_rolling_resamp.melt(ignore_index=False).reset_index()
#     vol_melt_rolling_resamp['rank'] = vol_melt_rolling_resamp.groupby(date)['value'].rank(method='min', ascending=False)
#     rank_raw = vol_melt_rolling_resamp.sort_values(by=['order_date', 'rank'], ascending=[True, True])
    
#     if rank_plot_on:
#         print('>>>> rank matrix')
#         plt = plot_rank_matrix(rank_raw)
#         plt.show
#     return rank_raw


# def transform_rank_fill_order_dates(last_order, rank, markets=None):
#     '''
#     this function
#     '''

#     # TODO
#     # check ranks markets
#     # warning if rank markets and markets does not match
#     # pivot rank
#     rank_pivot = rank.pivot_table(index='order_date', columns='market_maker_mnemonic', values='rank', aggfunc='mean')

#     # create a dataframe with only the dates
#     lo_dates = pd.DataFrame(index=last_order.order_date)

#     # join all teh dates in last_order
#     rank_merged = pd.merge(lo_dates, rank_pivot, how='left', left_index=True, right_index=True)

#     # first forward fill data, then backward fill data
#     # forward fill will use the latest resampled ranking from the dataset, e.g. end of previous month. This will prevent data leakage
#     # backward fill will fill the missing date's value
#     rank_merged = rank_merged.ffill().bfill()

#     # rank long = [index, market maker, rank] there is no market here any more
#     rank_filled = rank_merged.melt(ignore_index=False, var_name='market_maker_mnemonic', value_name='rank')

#     return rank_filled