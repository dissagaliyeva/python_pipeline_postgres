from data.helpers import *
from data.rank import *


def start_download(today_date, save_path, df_path, history=False, target_col='F380 M2/M3'):
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
    print('Last date:', last_date)

    # ============= RAW DATA =============

    # download new values
    raw_data = download_data(start_date=last_date, end_date=today_date, headers=HEADERS,
                             order_state='"active","consummated","inactive","withdrawn"')

    # standardize data
    raw_data = preprocess_data(raw_data, 'raw_data')
    append_pickle(raw_data, save_path, 'asia')

    # ============= LAST ORDER =============

    # get last order values
    last_order = get_last_order(raw_data)

    # standardize data
    last_order = preprocess_data(last_order, 'last_order')

    append_pickle(last_order, save_path, 'last_order')

    # ============= PVO EXPANDED =============

    # PVO expanded
    lo_expanded = preprocess_data(get_lo_vol_expanded(last_order), 'pvo')
    append_pickle(lo_expanded, save_path, 'lo_expanded')

    data = add_final_ranks(today_date, mdd, last_date, save_path, target_col)
    
    # store last 5 files, otherwise delete
    store_data(data, today_date)
    

def store_data(data, today_date):
    # store the latest file
    data.to_csv(f'data/data_{today_date}.csv')
    
    # remove files that are older than 5 days
    files = sorted(glob.iglob(f'data/data_*'), key=os.path.getctime, reverse=True)
    
    if len(files) > 5:
        for file in files[5:]:
            os.remove(file)
    
    

def add_final_ranks(today_date, mdd, last_date, save_path, target_col='F380 M2/M3'):
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

    # add ranking group name
    daily_ranking = add_rank_group(daily_ranking)

    if type(daily_ranking.index) != pd.core.indexes.range.RangeIndex:
        daily_ranking.reset_index(inplace=True)

    # path, file, filename
    # save_to_pickle(daily_ranking, 'daily_ranking', operation='replace')
    daily_ranking.to_csv(save_path + '/' + 'daily_ranking.csv')

    print('Daily Ranking:', daily_ranking)

    last_order_ranked = pd.merge(last_order, daily_ranking, on=['order_date', 'market_maker_mnemonic'], how='left')
    print('Last order ranked:', last_order_ranked)

    columns = ['order_date', 'market_maker_mnemonic', 'rank', 'rank_group', 'order_id']
    pvo_expanded = pd.merge(pvo_expanded, last_order_ranked[columns], on=['order_date', 'order_id'])
    pvo_expanded.to_csv('data/pvo_expanded.csv')
    
    pvo_expanded = pd.read_csv('data/pvo_expanded.csv', index_col=0)
    pvo_expanded.pricing_date = pd.to_datetime(pvo_expanded.pricing_date)


    ml_feed_mm = pvo_expanded.copy().pivot_table(index='pricing_date', columns=['market_maker_mnemonic'],
                                                 values='daily_quantity').fillna(0)
    ml_feed_rg = pvo_expanded.copy().pivot_table(index='pricing_date', columns=['rank_group'],
                                                 values='daily_bo_quantity', aggfunc=np.sum, ).fillna(0)
    ml_feed_agg = ml_feed_mm.join(ml_feed_rg)

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
    
    # TODO: append to existing location

    # save_to_pickle(save_path, ml_feed_mm, 'ml_feed_mm.csv')
    # save_to_pickle(save_path, ml_feed_rg, 'ml_feed_rg.csv')
    # save_to_pickle(save_path, ml_feed_agg, 'ml_feed_agg.csv')

    # ml_feed_mm.to_csv(save_path + '/' + 'ml_feed_mm.csv')
    # ml_feed_rg.to_csv(save_path + '/' + 'ml_feed_rg.csv')
    # ml_feed_agg.to_csv(save_path + '/' + 'ml_feed_agg.csv')

    return ml_feed_agg


def get_historic_data(filename):
    history = get_last_file(filename)
    
    if history is None:
        if 'pvo' in filename or 'lo_expanded' in filename:
            # load historic last_order and preprocess
            last_order = preprocess_data(pickle.load(open(get_last_file('last_order'), 'rb')), 'last_order')
            history = preprocess_data(get_lo_vol_expanded(last_order), 'pvo')
            history.to_pickle('Pickle/lo_vol_expanded.pickle')
            return history
        raise FileNotFoundError(f'Trying to load unknown file type {filename}')
    
    return preprocess_data(pickle.load(open(history, 'rb')), filename)
    


def get_last_file(keyword) -> str:
    # look for last file matching description
    latest_file = sorted(glob.iglob('Pickle/*'), key=os.path.getctime, reverse=True)
    latest = [x for x in latest_file if keyword in x]
    print('LATEST:', latest)
    
    if len(latest) > 0:
        return latest[0]
    
    return None


def concat_target(data, prices, target_col='F380 M2/M3'):
    data2, target2 = data.copy(), prices.copy()

    if data2.index.name != 'pricing_date':
        data2.set_index('pricing_date', inplace=True)

    data2[target_col] = target2[target_col]
    return data2


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


def merge_files(history, new_file, filetype='last_order'):
    # read in the file and preprocess
    history = preprocess_data(pickle.load(open(history, 'rb')), filetype)

    # append and preprocess files
    return preprocess_data(pd.concat([history, new_file]), filetype)


def save_to_pickle(path, file, filename):
    file.to_pickle(f'{path}/{filename}')




# from custom.data.helpers import *
# from custom.data.rank import *


# def start_download(today_date, save_path, df_path, history=False, target_col='F380 M2/M3'):
#     API_TOKEN = 'hsZhxtLPXCvgrIagPFpq'
#     HEADERS = {'appkey': API_TOKEN, 'Content-Type': 'application/json'}
    
#     # set credentials 
#     ci.set_credentials("simon.xin@mlp.com", "Godbless!001", "HJrqeVqsRsWEYqmxpGBU")
#     mdd = ci.MarketData()
    
    
#     # ======================================================================================= #
#     # ========================          PICKLE IMPLEMENTATION          ========================
#     # ======================================================================================= #

#     # check last available date
#     last_date = check_last_date() if history is False else '2017-01-01'
#     print('Last date:', last_date)

#     # ============= RAW DATA =============

#     # download new values
#     raw_data = download_data(start_date=last_date, end_date=today_date, headers=HEADERS,
#                              order_state='"active","consummated","inactive","withdrawn"')

#     # standardize data
#     raw_data = preprocess_data(raw_data, 'raw_data')
#     append_pickle(raw_data, save_path, 'asia')

#     # ============= LAST ORDER =============

#     # get last order values
#     last_order = get_last_order(raw_data)

#     # standardize data
#     last_order = preprocess_data(last_order, 'last_order')

#     append_pickle(last_order, save_path, 'last_order')

#     # ============= PVO EXPANDED =============

#     # PVO expanded
#     lo_expanded = preprocess_data(get_lo_vol_expanded(last_order), 'pvo')
#     append_pickle(lo_expanded, save_path, 'lo_expanded')

#     data = add_final_ranks(today_date, mdd, last_date, save_path, target_col)
#     data.to_csv(df_path)
    


# def add_final_ranks(today_date, mdd, last_date, save_path, target_col='F380 M2/M3'):
#     # ============= ADD RANKS =============
#     # get maximum date
#     # Assume we have the rnak table, from rank table in DB, latest date = last hist date, 180 days + buffer,
#     # e.g. rank based on 180 day moving average, resample the volume every Quarter,  01 Sep - 11 Oct, -180 --> df

#     # get the whole dataset from db
#     last_order = get_historic_data('last_order')
#     last_order.order_date = pd.to_datetime(last_order.order_date)
#     print('Last order head:', last_order.head())
#     print('Last order tail:', last_order.tail())

#     pvo_expanded = get_historic_data('lo_expanded')
#     pvo_expanded.order_date = pd.to_datetime(pvo_expanded.order_date)
#     print('lo_expanded head:', pvo_expanded.head())
#     print('lo_expanded tail:', pvo_expanded.tail())

#     # last_hist_date = check_last_date('daily_rank')
#     period_ranking = get_ranking(last_order, n=180, period='Q')
#     daily_ranking = transform_rank_fill_order_dates(last_order, period_ranking)

#     # add ranking group name
#     daily_ranking = add_rank_group(daily_ranking)

#     if type(daily_ranking.index) != pd.core.indexes.range.RangeIndex:
#         daily_ranking.reset_index(inplace=True)

#     # path, file, filename
#     # save_to_pickle(daily_ranking, 'daily_ranking', operation='replace')
#     daily_ranking.to_csv(save_path + '/' + 'daily_ranking.csv')

#     print('Daily Ranking:', daily_ranking)

#     last_order_ranked = pd.merge(last_order, daily_ranking, on=['order_date', 'market_maker_mnemonic'], how='left')
#     print('Last order ranked:', last_order_ranked)

#     columns = ['order_date', 'market_maker_mnemonic', 'rank', 'rank_group', 'order_id']
#     pvo_expanded = pd.merge(pvo_expanded, last_order_ranked[columns], on=['order_date', 'order_id'])
#     print('Pvo expanded:', pvo_expanded)

#     ml_feed_mm = pvo_expanded.copy().pivot_table(index='pricing_date', columns=['market_maker_mnemonic'],
#                                                  values='daily_quantity').fillna(0)
#     ml_feed_rg = pvo_expanded.copy().pivot_table(index='pricing_date', columns=['rank_group'],
#                                                  values='daily_bo_quantity', aggfunc=np.sum, ).fillna(0)
#     ml_feed_agg = ml_feed_mm.join(ml_feed_rg)

#     if target_col not in pvo_expanded.columns:
#         print('Last order head:', last_order.head())
#         last_order['order_date'] = pd.to_datetime(last_order['order_date'])
#         prices = get_prices(mdd, start_date='2017-01-01', end_date=today_date)
#     else:
#         prices = get_prices(mdd, start_date=last_date, end_date=today_date)

#     prices.index.name = 'pricing_date'
#     prices.sort_index(inplace=True)

#     # append target columns
#     ml_feed_mm = concat_target(ml_feed_mm, prices, target_col)
#     ml_feed_rg = concat_target(ml_feed_rg, prices, target_col)
#     ml_feed_agg = concat_target(ml_feed_agg, prices, target_col)

#     save_to_pickle(save_path, ml_feed_mm, 'ml_feed_mm.csv')
#     save_to_pickle(save_path, ml_feed_rg, 'ml_feed_rg.csv')
#     save_to_pickle(save_path, ml_feed_agg, 'ml_feed_agg.csv')

#     ml_feed_mm.to_csv(save_path + '/' + 'ml_feed_mm.csv')
#     ml_feed_rg.to_csv(save_path + '/' + 'ml_feed_rg.csv')
#     ml_feed_agg.to_csv(save_path + '/' + 'ml_feed_agg.csv')

#     return ml_feed_agg


# def get_historic_data(filename):
#     history = get_last_file(filename)
    
#     if history is None:
#         if 'pvo' in filename or 'lo_expanded' in filename:
#             # load historic last_order and preprocess
#             last_order = preprocess_data(pickle.load(open(get_last_file('last_order'), 'rb')), 'last_order')
#             history = preprocess_data(get_lo_vol_expanded(last_order), 'pvo')
#             history.to_pickle('Pickle/lo_vol_expanded.pickle')
#             return history
#         raise FileNotFoundError(f'Trying to load unknown file type {filename}')
    
#     return preprocess_data(pickle.load(open(history, 'rb')), filename)
    


# def get_last_file(keyword) -> str:
#     # look for last file matching description
#     latest_file = sorted(glob.iglob('Pickle/*'), key=os.path.getctime, reverse=True)
#     latest = [x for x in latest_file if keyword in x]
#     print('LATEST:', latest)
    
#     if len(latest) > 0:
#         return latest[0]
    
#     return None


# def concat_target(data, prices, target_col='F380 M2/M3'):
#     data2, target2 = data.copy(), prices.copy()

#     if data2.index.name != 'pricing_date':
#         data2.set_index('pricing_date', inplace=True)

#     data2[target_col] = target2[target_col]
#     return data2


# def get_raw_data(start_date, end_date, headers, order_state):
#     # download raw data
#     raw_data = download_data(start_date=start_date, end_date=end_date, headers=headers, order_state=order_state)
#     return preprocess_data(raw_data, 'raw_data')


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

#     last_order['oco_order_no'] = last_order['oco_order_id'].apply(
#         lambda x: 1 + round((len(x) + 1) / 9, 0) if x != None else 1)

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

#     return last_order


# def get_prices(mdd, start_date, end_date, target_col='F380 M2/M3'):
#     ric_dict = {
#         # Structure
#         'AAWGS00': 'F380 M1/M2',
#         'AAWGT00': 'F380 M2/M3',
#         'AAWGU00': 'F380 M3/M4',

#         # EW EU settlement
#         'FQLSM01': 'EW F380 vs F35 M1',
#         'FQLSM02': 'EW F380 vs F35 M2',
#         'FQLSM03': 'EW F380 vs F35 M3',

#         # Visco
#         'AAVFL00': 'SG Visco M1',
#         'AAVFM00': 'SG Visco M2',
#         'AAVFN00': 'SG Visco M3',

#         # Cracks
#         'AAWHA00': 'F380 DB Cracks M1',
#         'AAWHB00': 'F380 DB Cracks M2',

#         'ABWDN00': 'ULSD NWE M1',
#         'ABWDO00': 'ULSD NWE M2',
#         'ABWDP00': 'ULSD NWE M3',
#         'ABWDQ00': 'ULSD NWE M4',

#         # Middle Distillate
#         'AAXAL00': 'ULSD NWE vs ICEG M1',
#         'AAXAM00': 'ULSD NWE vs ICEG M2'
#     }

#     curves = download_curves_spgi(mdd, ric_dict, start_date=start_date, end_date=end_date)
#     return curves


# def merge_files(history, new_file, filetype='last_order'):
#     # read in the file and preprocess
#     history = preprocess_data(pickle.load(open(history, 'rb')), filetype)

#     # append and preprocess files
#     return preprocess_data(pd.concat([history, new_file]), filetype)


# def save_to_pickle(path, file, filename):
#     file.to_pickle(f'{path}/{filename}')



# # import os.path
# # import pickle

# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import MinMaxScaler

# # from datetime import datetime
# # import spgci as ci

# # from airflow_utils.data import download_data

# # CRITICAL = 'CRITICAL'


# # def check_data():
# #     today = 'data/data_' + str(datetime.today())[:10] + '.csv'

# #     # read in existing data
# #     if os.path.exists(today):
# #         print('Loading existing data...')
# #         data = pd.read_csv(today)
# #         data.pricing_date = pd.to_datetime(data.pricing_date)
# #         data.set_index('pricing_date', inplace=True)
# #         return data

# #     # otherwise, download new data
# #     print('Starting to download!')
# #     data = download_data.start_download(str(datetime.today())[:10])
# #     data.to_csv(today)
# #     return data


# # def prepare_data(data, log, log_path, train_start, train_end, test_start, test_end=None,
# #                  target='F380 M2/M3', results_folder=None, variance=0.3):

# #     data.reset_index(inplace=True)
# #     data.drop(columns='index', inplace=True)
# #     data.pricing_date = pd.to_datetime(data.pricing_date)
# #     data.set_index('pricing_date', inplace=True)

# #     # select end date for test set if not specified
# #     if test_end is None:
# #         test_end = pd.DataFrame(data[data.index >= test_start].iloc[10, :]).T.index.values[0]
# #         log(log_path, f'Task 2: Assigning test_end date to {test_end}')

# #     print('Test end:', test_end)
# #     print(data)

# #     print('Test start:', test_start)
# #     print('Test end:', test_end)

# #     print('data[(data.index >= train_start) & (data.index <= test_end)]:', data[(data.index >= train_start) & (data.index <= test_end)])

# #     # select start and end dates
# #     data = data[(data.index >= train_start) & (data.index <= test_end)]
# #     data[target] = data[target].fillna(0)

# #     print(data)
# #     data.to_csv('prepare_data.csv')

# #     log(log_path, f'Slicing the data to start from {train_start} and end with {test_end} dates.')

# #     # drop missing columns
# #     old_shape = data.shape[0]
# #     data.dropna(inplace=True)
# #     log(log_path, f'Dropped missing columns. Number of missing columns: {data.shape[0] - old_shape}.')

# #     # separate target and feature columns
# #     target_feature = TargetFeature(log_path, log, results_path=results_folder,
# #                                    data=data, test_start=test_start, test_end=test_end)
# #     target_feature()
# #     log(log_path, 'Created TargetFeature class.')

# #     # drop target column
# #     data.drop(columns=target, inplace=True)

# #     # add time-related features if not present
# #     data = add_time_features(data, log=log, log_path=log_path)

# #     # get columns with variance bigger than the specified variance (default=0.2)
# #     columns = drop_low_variance(data, test_start, log, log_path, test_end, var=variance)
# #     log(log_path, f'Selected {len(columns)} with variance={variance}: {columns}.')

# #     # normalize data
# #     train, test = data.loc[:test_start, :][columns], data.loc[test_start:, :][columns]
# #     feature_scaler = MinMaxScaler().fit(train)

# #     train = pd.DataFrame(feature_scaler.transform(train), index=train.index, columns=train.columns)
# #     test = pd.DataFrame(feature_scaler.transform(test), index=test.index, columns=test.columns)

# #     # final log
# #     log(log_path, f'Task 2: Completed. Returning data, columns, and TargetFeature class instance.')

# #     # save current state in a pickle
# #     pickle_loc = f'{results_folder}/pickles'
# #     pickle_file = f'{pickle_loc}/results_variance={variance}'
    
# #     print('PICKLE FILE CREATED:', pickle_file)

# #     # create a folder to store results
# #     if not os.path.exists(pickle_loc): os.mkdir(pickle_loc)

# #     # store results
# #     with open(pickle_file, 'wb') as fout:
# #         pickle.dump({'data': data, 'columns': columns.to_list(), 'target_feature': target_feature,
# #                      'test_end': test_end, 'train': train, 'test': test, 'feature_scaler': feature_scaler},
# #                     fout, pickle.HIGHEST_PROTOCOL)

# #     return pickle_file


# # def get_data(add_dow=False, old_data=True):
# #     if old_data:
# #         data = pd.read_csv('data/agg_data_daily_2023-03-11_12-31-45.csv')
# #     else:
# #         data = pd.read_csv('data/agg_data_daily_2023-05-02_16-29-18.csv')

# #     data.pricing_date = pd.to_datetime(data.pricing_date)

# #     data.set_index('pricing_date', inplace=True)

# #     data['year_sin'] = np.sin(data.index.year / data.index.year.max() * 2 * np.pi)

# #     data['month_sin'] = np.sin(data.index.month / data.index.month.max() * 2 * np.pi)

# #     data['day_sin'] = np.sin(data.index.day / data.index.day.max() * 2 * np.pi)

# #     data['dow_sin'] = np.sin(data.index.dayofweek / data.index.dayofweek.max() * 2 * np.pi)

# #     if old_data:
# #         if add_dow:
# #             data = data[[*data.columns[24:-8], *data.columns[-5:]]]
# #         else:
# #             data = data[[*data.columns[24:-7], *data.columns[-4:]]]

# #     return data


# # def get_combined_data(rgp=True):
# #     data1, data2 = get_data(True, old_data=True), get_data(True, old_data=False)
# #     df = pd.concat([data2, data1.iloc[:, :-5]], axis=1)

# #     if rgp:
# #         cols = [x for x in df.columns.values if not x.startswith('_itr')]
# #         df = df[cols]

# #     return df


# # def add_time_features(data, log, log_path):
# #     log(log_path, 'Task 2: Adding time-related features...')

# #     if 'year_sin' not in data.columns:
# #         data['year_sin'] = np.sin(data.index.year / data.index.year.max() * 2 * np.pi)
# #         log(log_path, 'Added `year_sin` feature.')

# #     if 'month_sin' not in data.columns:
# #         data['month_sin'] = np.sin(data.index.month / data.index.month.max() * 2 * np.pi)
# #         log(log_path, 'Added `month_sin` feature.')

# #     if 'day_sin' not in data.columns:
# #         data['day_sin'] = np.sin(data.index.day / data.index.day.max() * 2 * np.pi)
# #         log(log_path, 'Added `day_sin` feature.')

# #     if 'dow_sin' not in data.columns:
# #         data['dow_sin'] = np.sin(data.index.dayofweek / data.index.dayofweek.max() * 2 * np.pi)
# #         log(log_path, 'Added `dow_sin` feature.')

# #     log(log_path, 'Task 2: Completed adding time-related features.')
# #     return data


# # def drop_low_variance(data, test_start, log, log_path, test_end, var=0.2):
# #     # log information
# #     log(log_path, f'Task 2: Selecting and dropping features with low variance (var < {var})...')

# #     # remove rows after test start
# #     data = data[(data.index > '2017-03-21') & (data.index < test_start)]

# #     # check the length of prediction dates
# #     if len(data[(data.index >= test_start) & (data.index < test_end)]) < 100:
# #         return data.columns[data.var() > var]

# #     return data[['_rgp_01_sz_3', '_rgp_02_sz_3', '_rgp_03_sz_3', '_rgp_04_sz_3', '_rgp_05_sz_3',
# #                  'BPSG', 'COASTAL', 'GUNVORSG', 'HL', 'MERCURIASG', 'P66SG', 'PETROCHINA',
# #                  'SIETCO', 'TOTALSG', 'TRAFI', 'VITOLSG']].columns


# # class TargetFeature:
# #     def __init__(self, log_path, log, results_path, data, test_start, test_end, target='F380 M2/M3',
# #                  feature='rolling_target5', **kwargs):
# #         self.log = log
# #         self.log_path = log_path
# #         self.results_path = results_path

# #         self.data = data
# #         self.test_start = test_start
# #         self.test_end = test_end
# #         self.target = target
# #         self.feature = feature

# #         # define pd.DataFrame to store resulting df
# #         self.target_and_feature = None

# #         # define scaler
# #         self.scaler = MinMaxScaler()

# #         # define a starting message for log
# #         self.message = 'Task 2 [TargetFeature]:\n\t'

# #     def __call__(self, *args, **kwargs):
# #         self._setup()

# #     def _setup(self):
# #         self.log(self.log_path, f'{self.message}Internal setup for target and feature columns...')

# #         # get rolling feature
# #         rolling = self.data[self.target].rolling(5).mean()
# #         rolling.dropna(inplace=True)
# #         self.log(self.log_path, f'Created rolling target.')

# #         # fit a scaler
# #         self.scaler.fit(rolling[:self.test_end].values.reshape(-1, 1))
# #         self.log(self.log_path, f'Created a MinMaxScaler for rolling column.')

# #         # create a dataframe to store current state
# #         self.target_and_feature = pd.DataFrame({
# #             'actual_target': self.data[self.target],
# #             'actual_feature': rolling[:self.test_end],
# #             'predicted_target': np.nan,
# #             'predicted_feature': np.nan
# #         }, index=self.data.index)
# #         self.log(self.log_path, f'Created a dataframe to store initial actual/predicted target and feature.')

# #         # fill in predicted target and feature until test_start date
# #         self.target_and_feature.loc[:self.test_start, 'predicted_target'] = self.data[self.target][:self.test_start]
# #         self.target_and_feature.loc[:self.test_start, 'predicted_feature'] = rolling[:self.test_start]
# #         self.log(self.log_path, f'Setting previous predicted target/feature columns to actual values.')
# #         self.log(self.log_path, f'{self.message}Finish setting up TargetFeature class instance.')

# #     def get_target_feature(self, end_date, start_date=None, include_past=False):
# #         # return all values until specified date
# #         if include_past:
# #             if start_date is not None:
# #                 self.log(self.log_path, f'Returning dates from {start_date} to {end_date}.')
# #                 return self.target_and_feature.loc[start_date:end_date, :]
# #             self.log(self.log_path, f'Returning all dates until {end_date}')
# #             return self.target_and_feature.loc[:end_date, :]

# #         # return one-day results
# #         try:
# #             entry = pd.DataFrame(self.target_and_feature.loc[end_date, :]).T
# #         except KeyError:
# #             self.log(self.log_path, f'{self.message}Date {end_date} does not exist!', CRITICAL)
# #             raise KeyError(f'{self.message}Date {end_date} does not exist!')
# #         else:
# #             self.log(self.log_path, f'Returned one-day instance for {end_date}.')
# #             return entry

# #     def set_target_feature(self, date, column, value):
# #         fmt = '%Y-%m-%d'

# #         # raise error if the supplemented date is less than the test start
# #         # if datetime.strptime(self.test_start, fmt) < datetime.strptime(date, fmt):
# #         #     message = f'Illegal date substitution {date}, expected to see dates between {self.test_start} to {date}.'
# #         #     self.log(self.log_path, self.message + message, CRITICAL)
# #         #     raise Exception(message)

# #         # get previous value
# #         previous = self.target_and_feature.loc[date, column]

# #         # log information
# #         self.log(self.log_path, f'{self.message}Setting new value for {column}. Old: {previous}, new: {value}')

# #         # set new value
# #         self.target_and_feature.loc[date, column] = value

# #         # save current state
# #         # self.target_and_feature.reset_index(inplace=True)
# #         self.target_and_feature.to_csv(f'{self.results_path}/complete_target_feature_df.csv', index=True)
# #         self.log(self.log_path, f'{self.message}Saving intermediate results in {self.results_path} CSV file.')
