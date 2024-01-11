from helpers import *
from rank import *


def start_download(today_date, history=False, target_col='F380 M2/M3'):
    API_TOKEN = 'hsZhxtLPXCvgrIagPFpq'
    HEADERS = {'appkey': API_TOKEN, 'Content-Type': 'application/json'}

    # set credentials
    ci.set_credentials("simon.xin@mlp.com", "Godbless!001", "HJrqeVqsRsWEYqmxpGBU")
    mdd = ci.MarketData()

    # ======================================================================================= #
    # ========================          PICKLE IMPLEMENTATION          ========================
    # ======================================================================================= #

    # check last available date
    last_date = check_last_date() if history is False else '2017-01-01'

    # ============= RAW DATA =============

    # download new values
    raw_data = download_data(start_date=last_date, end_date=today_date, headers=HEADERS,
                             order_state='"active","consummated","inactive","withdrawn"')

    # standardize data
    raw_data = preprocess_data(raw_data, 'raw_data')
    add_to_database(raw_data, 'raw_order')

    # ============= LAST ORDER =============

    # get last order values
    last_order = get_last_order(raw_data)

    # standardize data
    last_order = preprocess_data(last_order, 'last_order')

    add_to_database(last_order, 'last_order')

    # ============= PVO EXPANDED =============

    # PVO expanded
    lo_expanded = preprocess_data(get_lo_vol_expanded(last_order), 'pvo')
    add_to_database(lo_expanded, 'lo_expanded')

    return add_final_ranks(today_date, mdd, last_date)


def add_final_ranks(today_date, mdd, last_date, target_col='F380 M2/M3'):
    # ============= ADD RANKS =============
    # get maximum date
    # Assume we have the rnak table, from rank table in DB, latest date = last hist date, 180 days + buffer,
    # e.g. rank based on 180 day moving average, resample the volume every Quarter,  01 Sep - 11 Oct, -180 --> df

    # get the whole dataset from db
    last_order = get_historic_data('last_order')
    last_order.order_date = pd.to_datetime(last_order.order_date)
    print('Last order head:', last_order.head())
    print('Last order tail:', last_order.tail())

    pvo_expanded = get_historic_data('lo_expanded')
    pvo_expanded.order_date = pd.to_datetime(pvo_expanded.order_date)
    print('lo_expanded head:', pvo_expanded.head())
    print('lo_expanded tail:', pvo_expanded.tail())

    # last_hist_date = check_last_date('daily_rank')
    period_ranking = get_ranking(last_order, n=180, period='Q')
    daily_ranking = transform_rank_fill_order_dates(last_order, period_ranking)

    # last_order.order_date = pd.to_datetime(last_order.order_date)

    # if last_hist_date is None:
    #     period_ranking = get_ranking(last_order, n=180, period='Q')
    #     daily_ranking = transform_rank_fill_order_dates(last_order, period_ranking)
    # else:
    #     # TODO: FIX THE PIPELINE
    #     period_ranking = get_ranking(last_order[last_order.order_date >= last_hist_date], n=180, period='Q')
    #     daily_ranking = transform_rank_fill_order_dates(last_order, period_ranking)

    # add ranking group name
    daily_ranking = add_rank_group(daily_ranking)

    if type(daily_ranking.index) != pd.core.indexes.range.RangeIndex:
        daily_ranking.reset_index(inplace=True)

    add_to_database(daily_ranking, 'daily_ranking', operation='replace')

    print('Daily Ranking:', daily_ranking)

    last_order_ranked = pd.merge(last_order, daily_ranking, on=['order_date', 'market_maker_mnemonic'], how='left')
    print('Last order ranked:', last_order_ranked)

    columns = ['order_date', 'market_maker_mnemonic', 'rank', 'rank_group', 'order_id']
    pvo_expanded = pd.merge(pvo_expanded, last_order_ranked[columns], on=['order_date', 'order_id'])
    print('Pvo expanded:', pvo_expanded)

    ml_feed_mm = pvo_expanded.copy().pivot_table(index='pricing_date', columns=['market_maker_mnemonic'],
                                                 values='daily_quantity').fillna(0)
    ml_feed_rg = pvo_expanded.copy().pivot_table(index='pricing_date', columns=['rank_group'],
                                                 values='daily_bo_quantity', aggfunc=np.sum, ).fillna(0)
    ml_feed_agg = ml_feed_mm.join(ml_feed_rg)

    ml_feed_mm.to_csv('ml_feed_mm.csv')
    ml_feed_rg.to_csv('ml_feed_rg.csv')
    ml_feed_agg.to_csv('ml_feed_agg.csv')

    if target_col not in pvo_expanded.columns:
        print('Last order head:', last_order.head())
        last_order['order_date'] = pd.to_datetime(last_order['order_date'])
        prices = get_prices(mdd, start_date='2017-01-01', end_date=today_date)
    else:
        prices = get_prices(mdd, start_date=last_date, end_date=today_date)

    prices.index.name = 'pricing_date'
    prices.sort_index(inplace=True)

    # append target columns
    ml_feed_mm = concat_target(ml_feed_mm, prices, target_col)
    ml_feed_rg = concat_target(ml_feed_rg, prices, target_col)
    ml_feed_agg = concat_target(ml_feed_agg, prices, target_col)

    add_to_database(ml_feed_mm, 'ml_feed_mm', operation='replace')
    add_to_database(ml_feed_rg, 'ml_feed_rg', operation='replace')
    add_to_database(ml_feed_agg, 'ml_feed_agg', operation='replace')

    return ml_feed_mm


def get_historic_data(table_name):
    # connect to the database
    con = sqlite3.connect('airflow_pipeline.db')

    # read in complete data
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', con)
    df.drop_duplicates(inplace=True)

    con.close()

    return df


def concat_target(data, prices, target_col='F380 M2/M3'):
    data2, target2 = data.copy(), prices.copy()

    # print('Current data head:', data.head())
    # print('Current target head:', prices.head())

    if data2.index.name != 'pricing_date':
        data2.set_index('pricing_date', inplace=True)
    # elif target2.index.name != 'pricing_date':
    #     target2.set_index('pricing_date', inplace=True)

    # print('Current data head:', data.head())
    # print('Current target head:', prices.head())

    # target2.index = pd.to_datetime(target2.index)
    # data2.index = pd.to_datetime(data2.index)

    # target2.reset_index(inplace=True)
    # data2.reset_index(inplace=True)

    data2[target_col] = target2[target_col]
    return data2
    # return pd.concat([data2, prices[[target_col]]], ignore_index=False, axis=1)


def add_to_database(data, filename, operation='append'):
    # connect to the database
    con = sqlite3.connect('airflow_pipeline.db')

    if 'order_date' in data.columns:
        data = data.sort_values('order_date').drop_duplicates()

    if data.index.name == 'pricing_date':
        data.reset_index(inplace=True)

    data.to_sql(filename, con, if_exists=operation, index=False)

    con.close()


def get_raw_data(start_date, end_date, headers, order_state):
    # download raw data
    raw_data = download_data(start_date=start_date, end_date=end_date, headers=headers, order_state=order_state)
    return preprocess_data(raw_data, 'raw_data')


def get_last_order(raw_data):
    last_order = raw_data[['order_time', 'order_type', 'order_state', 'update_time', 'market',
                           'market_type', 'order_id', 'oco_order_id', 'deal_id', 'market_maker',
                           'market_maker_mnemonic',
                           'order_date', 'order_begin', 'order_end',
                           'order_quantity', 'order_cancelled',
                           'product', 'hub', 'strip', 'c1_basis_period', 'c1_basis_period_details', 'c1_price',
                           'c1_price_basis', ]]
    # sort and keep last
    last_order = last_order.sort_values('order_time', ascending=True).drop_duplicates(['order_date', 'order_id'],
                                                                                      keep='last')
    # add new column with volume bid = positive offer = negative
    last_order['bo_quantity'] = np.where((last_order['order_type'] == 'Offer'), last_order['order_quantity'] * -1,
                                         last_order['order_quantity'])

    last_order['oco_order_no'] = last_order['oco_order_id'].apply(
        lambda x: 1 + round((len(x) + 1) / 9, 0) if x != None else 1)

    # TODO: add 1 day to order_end (just a day not business day)
    one_more_day = pd.DatetimeIndex(last_order.order_end.values.astype('datetime64[D]')) + pd.DateOffset(1)
    last_order['pricing_days'] = np.busday_count(last_order.order_begin.values.astype('datetime64[D]'),
                                                 one_more_day.values.astype('datetime64[D]'))
    # #divide the total volume by number of oco contracts, if no oco, divide by 1
    last_order['daily_quantity'] = last_order['order_quantity'] / (
            last_order['pricing_days'] * last_order['oco_order_no'])
    last_order['daily_bo_quantity'] = last_order['bo_quantity'] / (
            last_order['pricing_days'] * last_order['oco_order_no'])
    last_order['c1_price_basis'] = last_order['c1_price_basis'].str.rstrip('.')

    return last_order


def get_prices(mdd, start_date, end_date, target_col='F380 M2/M3'):
    ric_dict = {
        # Structure
        'AAWGS00': 'F380 M1/M2',
        'AAWGT00': 'F380 M2/M3',
        'AAWGU00': 'F380 M3/M4',

        # EW EU settlement
        'FQLSM01': 'EW F380 vs F35 M1',
        'FQLSM02': 'EW F380 vs F35 M2',
        'FQLSM03': 'EW F380 vs F35 M3',

        # Visco
        'AAVFL00': 'SG Visco M1',
        'AAVFM00': 'SG Visco M2',
        'AAVFN00': 'SG Visco M3',

        # Cracks
        'AAWHA00': 'F380 DB Cracks M1',
        'AAWHB00': 'F380 DB Cracks M2',

        'ABWDN00': 'ULSD NWE M1',
        'ABWDO00': 'ULSD NWE M2',
        'ABWDP00': 'ULSD NWE M3',
        'ABWDQ00': 'ULSD NWE M4',

        # Middle Distillate
        'AAXAL00': 'ULSD NWE vs ICEG M1',
        'AAXAM00': 'ULSD NWE vs ICEG M2'
    }

    curves = download_curves_spgi(mdd, ric_dict, start_date=start_date, end_date=end_date)
    return curves


# def start_historical_merge(last_order, expanded, date, volume=True):
#     # load latest last_order file
#     last_order_history = get_last_file('last_order')
#
#     # load latest expanded (lo_expanded or lo_vol_expanded)
#     expanded_history = get_last_file('lo_expanded') if not volume else get_last_file('lo_vol_expanded')
#
#     # check for duplicates and overlapping
#     last_order_merged = merge_files(last_order_history, last_order)
#     expanded_merged = merge_files(expanded_history, expanded)
#
#     # save files
#     save_to_pickle(last_order_merged, f'last_order_{date}')

    # ranks will be merged in later. NO RANK
    # check if there is recent enough ranks to use
    # merge and save it


def merge_files(history, new_file, filetype='last_order'):
    # read in the file and preprocess
    history = preprocess_data(pickle.load(open(history, 'rb')), filetype)

    # append and preprocess files
    return preprocess_data(pd.concat([history, new_file]), filetype)


def save_to_pickle(file, filename):
    file.to_pickle(f'Pickle/{filename}')



# from helpers import *
#
#
# def main(today_date=None, start_date=None, end_date=None, download_history=False, volume=False):
#     # get today's date
#     today_date = datetime.today() if today_date is None else today_date
#
#     # set headers and order state
#     headers, mdd = get_headers()
#     order_state = '"active","consummated","inactive","withdrawn"'
#
#     # if all three values are supplemented, start downloading historic data
#     if download_history and start_date and end_date:
#         dt_format = '%Y-%m-%d'
#
#         assert datetime.strptime(start_date, dt_format) < datetime.strptime(end_date, dt_format), \
#             f'Illegal start date substitution (start date >= end_date): start_date={start_date}, end_date={end_date}'
#         assert datetime.strptime(end_date, dt_format) <= datetime.today(), \
#             f'End date cannot be bigger than today\'s date: end_date={end_date}'
#
#     else:
#         # otherwise, get the last order date available from lo_vol_expanded or last_order date
#         start_date = check_last_date()
#         end_date = today_date
#
#     # download raw data
#     raw_data = preprocess_data(get_raw_data(start_date, end_date, headers, order_state), 'raw_data')
#
#     # download last order data
#     last_order = preprocess_data(get_last_order(raw_data), 'last_order')
#
#     if volume:
#         lo_vol_expanded = preprocess_data(get_lo_vol_expanded(last_order), 'last_order')
#         start_historical_merge(last_order, lo_vol_expanded, today_date, volume=True)
#     else:
#         lo_expanded = preprocess_data(get_lo_expanded(last_order), 'last_order')
#         start_historical_merge(last_order, lo_expanded, today_date, volume=False)
#
#     # get prices
#     prices = get_prices(mdd, start_date, end_date)
#
#     # add ranks
#     start_fill_rank(last_order, 1)
#
#     # combine everything together
#
#     # add target column
#
#     # save everything
#
#     # return formatted data
#
#
# def get_raw_data(start_date, end_date, headers, order_state):
#     # download raw data
#     raw_data = download_data(start_date=start_date, end_date=end_date, headers=headers, order_state=order_state)
#     return preprocess_data(raw_data, 'raw_data')
#
#
# def get_last_order(raw_data):
#     last_order = raw_data[['order_time', 'order_type', 'order_state', 'update_time', 'market',
#                            'market_type', 'order_id', 'oco_order_id', 'deal_id', 'market_maker',
#                            'market_maker_mnemonic',
#                            'order_date', 'order_begin', 'order_end',
#                            'order_quantity', 'order_cancelled',
#                            'product', 'hub', 'strip', 'c1_basis_period', 'c1_basis_period_details', 'c1_price',
#                            'c1_price_basis', ]]
#     # sort and keep last
#     last_order = last_order.sort_values('order_time', ascending=True).drop_duplicates(['order_date', 'order_id'],
#                                                                                       keep='last')
#     # add new column with volume bid = positive offer = negative
#     last_order['bo_quantity'] = np.where((last_order['order_type'] == 'Offer'), last_order['order_quantity'] * -1,
#                                          last_order['order_quantity'])
#
#     last_order['oco_order_no'] = last_order['oco_order_id'].apply(
#         lambda x: 1 + round((len(x) + 1) / 9, 0) if x != None else 1)
#
#     # TODO: add 1 day to order_end (just a day not business day)
#     one_more_day = pd.DatetimeIndex(last_order.order_end.values.astype('datetime64[D]')) + pd.DateOffset(1)
#     last_order['pricing_days'] = np.busday_count(last_order.order_begin.values.astype('datetime64[D]'),
#                                                  one_more_day.values.astype('datetime64[D]'))
#     # #divide the total volume by number of oco contracts, if no oco, divide by 1
#     last_order['daily_quantity'] = last_order['order_quantity'] / (
#             last_order['pricing_days'] * last_order['oco_order_no'])
#     last_order['daily_bo_quantity'] = last_order['bo_quantity'] / (
#             last_order['pricing_days'] * last_order['oco_order_no'])
#     last_order['c1_price_basis'] = last_order['c1_price_basis'].str.rstrip('.')
#
#     return last_order
#
#
# def get_prices(mdd, start_date, end_date, target_col='F380 M2/M3'):
#     ric_dict = {
#                 # Structure
#                 'AAWGS00': 'F380 M1/M2',
#                 'AAWGT00': 'F380 M2/M3',
#                 'AAWGU00': 'F380 M3/M4',
#
#                 # EW EU settlement
#                 'FQLSM01': 'EW F380 vs F35 M1',
#                 'FQLSM02': 'EW F380 vs F35 M2',
#                 'FQLSM03': 'EW F380 vs F35 M3',
#
#                 # Visco
#                 'AAVFL00': 'SG Visco M1',
#                 'AAVFM00': 'SG Visco M2',
#                 'AAVFN00': 'SG Visco M3',
#
#                 # Cracks
#                 'AAWHA00': 'F380 DB Cracks M1',
#                 'AAWHB00': 'F380 DB Cracks M2',
#
#                 'ABWDN00': 'ULSD NWE M1',
#                 'ABWDO00': 'ULSD NWE M2',
#                 'ABWDP00': 'ULSD NWE M3',
#                 'ABWDQ00': 'ULSD NWE M4',
#
#                 # Middle Distillate
#                 'AAXAL00': 'ULSD NWE vs ICEG M1',
#                 'AAXAM00': 'ULSD NWE vs ICEG M2'
#             }
#
#     curves = download_curves_spgi(mdd, ric_dict, start_date=start_date, end_date=end_date)
#     curves.to_csv('prices.csv')
#     return curves[target_col].values
#
#
# def start_historical_merge(last_order, expanded, date, volume=True):
#     # load latest last_order file
#     last_order_history = get_last_file('last_order')
#
#     # load latest expanded (lo_expanded or lo_vol_expanded)
#     expanded_history = get_last_file('lo_expanded') if not volume else get_last_file('lo_vol_expanded')
#
#     # check for duplicates and overlapping
#     last_order_merged = merge_files(last_order_history, last_order)
#     expanded_merged = merge_files(expanded_history, expanded)
#
#     # save files
#     save_to_pickle(last_order_merged, f'last_order_{date}')
#
#     # ranks will be merged in later. NO RANK
#     # check if there is recent enough ranks to use
#     # merge and save it
#
#
# def merge_files(history, new_file, filetype='last_order'):
#     # read in the file and preprocess
#     history = preprocess_data(pickle.load(open(history, 'rb')), filetype)
#
#     # append and preprocess files
#     return preprocess_data(pd.concat([history, new_file]), filetype)
#
#
# def save_to_pickle(file, filename):
#     file.to_pickle(f'../../Pickle/{filename}')
#
#
# if __name__ == '__main__':
#     main(today_date='2023-09-20', download_history=False, volume=True)
