import pandas as pd
import spgci as ci

import os
import requests
import pickle
import time
import glob
import datetime as dt
from pathlib import Path
import sqlite3

import numpy as np
from pandas.tseries.offsets import BDay

from urllib.parse import quote
from datetime import timedelta, datetime
from pandas.tseries.offsets import *

import sys
print()


def loggg(f):
    def wrapper(*args, **kwargs):
        tic = datetime.now()
        result = f(*args, **kwargs)
        toc = datetime.now()
        print(f"{f.__name__} took {toc - tic} shape = {result.shape}")
        return result

    return wrapper


def append_target(features, target, history=False):
    # rename column
    target.index.name = 'pricing_date'

    # convert to datetime
    target.index = pd.to_datetime(target.index)

    features = features.join(target[['F380 M2/M3']], on='pricing_date')
    features.dropna(inplace=True)

    # reset index and drop duplicates
    features = features.reset_index().drop_duplicates('pricing_date').set_index('pricing_date')

    # concat with old values
    if history:
        # TO-DO: create target columns
        old = pd.read_csv('data/data-2023-08-25.csv')
        old.pricing_date = pd.to_datetime(old.pricing_date)
        old.set_index('pricing_date', inplace=True)

        return pd.concat([old, features], axis=0).reset_index().drop_duplicates('pricing_date')

    return features


def alternate_pivot(data):
    # Get unique values for index and columns
    index_vals = data['assessDate'].unique()
    col_vals = data['symbol'].unique()

    # Create an empty DataFrame to store the pivot result
    result = pd.DataFrame(index=index_vals, columns=col_vals)

    # Fill the result DataFrame using a loop
    for index_value in index_vals:
        for column_value in col_vals:
            value = data[(data['assessDate'] == index_value) & (data['symbol'] == column_value)]['value'].mean()
            result.at[index_value, column_value] = value

    # Convert the result to numeric (optional, if not already numeric)
    result = result.apply(pd.to_numeric)
    return result


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
    curves.to_csv('prices.csv')
    return curves


def download_curves_spgi(mdd, ric_dict, start_date='2017-01-01', end_date=dt.datetime.today()):
    # download data
    df = mdd.get_assessments_by_symbol_historical(
        symbol=list(ric_dict.keys()),
        bate=['c'],
        assess_date_gte=start_date,
        assess_date_lte=end_date,
        paginate=True,
    )

    # pivot and rename columns
    df = alternate_pivot(df)

    if len(ric_dict) > 1:
        df = df.rename(columns=ric_dict)

    return df


def format_data(pvo_expanded) -> pd.DataFrame:
    # get company names list
    columns = list(set(pvo_expanded['market_maker_mnemonic'].to_list()))
    columns += list(set(pvo_expanded['rank_group'].to_list()))

    # get dataframe with the values
    company_names = pvo_expanded.pivot_table(columns=pvo_expanded['market_maker_mnemonic'],
                                             index=pvo_expanded.pricing_date,
                                             values=pvo_expanded[['daily_quantity']])

    # get rgp groups
    rgp_groups = pvo_expanded.pivot_table(columns=pvo_expanded['rank_group'],
                                          index=pvo_expanded.pricing_date,
                                          values=pvo_expanded[['daily_quantity']])

    # combine datasets
    combined = pd.concat([company_names, rgp_groups])
    combined.columns = columns
    combined.fillna(0, inplace=True)

    return combined


def append_pickle(new_data, path, file='asia'):
    # get pickle file location
    if file == 'asia':
        file = file.title()
    if file.startswith('lo'): file = 'pvo_expanded'

    pickle_file = get_filename(file)
    print('Pickle file:', pickle_file)

    # open pickle file
    all_data = pickle.load(open(pickle_file, 'rb'))

    if file == 'Asia':
        if 'market_short_code' in all_data.columns:
            all_data['market_short_code'] = all_data['market_short_code'].apply(lambda x: x[0])

    # standardize data
    print('FILE:', file)
    print('All data:', all_data)
    ftype = 'raw_data' if file == 'Asia' else 'pvo'
    all_data = preprocess_data(all_data, ftype)

    # concat records
    combined = pd.concat([all_data, new_data], axis=0)
    
    # remove previous pickles
    remove_previous_pickles(file)
    
    # save to pickle
    combined.to_pickle(f'Pickle/' + file + str(dt.datetime.now())[:10] + '.pickle')

    return combined


def get_filename(fname):
    files = sorted(glob.iglob(f'Pickle/{fname}*'), key=os.path.getctime, reverse=True)
    print(f'[helpers.py @194] CURRENT EXISTING FILES FOR {fname}:', files)
    
    if len(files) == 1:
        if 'asia' in fname.lower():
            return 'Pickle/Asia_EU_FO_PVO_2016-2023.pickle'
        return f'Pickle/{fname}-2017-01-01-2023-10-09.pickle'
    
    return files[0]


def remove_previous_pickles(fname):
    files = sorted(glob.iglob(f'Pickle/{fname}*'), key=os.path.getctime, reverse=True)

    for file in files:
        if '2016-' in file or '2017-01-01' in file: continue
        os.remove(file)


def get_last_order(symbol_data):
    last_order = symbol_data[['order_time', 'order_type', 'order_state', 'update_time', 'market',
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
    last_order['pricing_days'] = np.busday_count(last_order.order_begin.values.astype('datetime64[D]'),
                                                 last_order.order_end.values.astype('datetime64[D]'))
    # #divide the total volume by number of oco contracts, if no oco, divide by 1
    last_order['daily_quantity'] = last_order['order_quantity'] / (
            last_order['pricing_days'] * last_order['oco_order_no'])
    last_order['daily_bo_quantity'] = last_order['bo_quantity'] / (
            last_order['pricing_days'] * last_order['oco_order_no'])
    last_order['c1_price_basis'] = last_order['c1_price_basis'].str.rstrip('.')

    return last_order



def get_lo_vol_expanded(last_order):
    expand_pvo_order = pd.DataFrame()

    for index, r in last_order.iterrows():
        # only expand phyiscal order
        if r.market_type == 'Platts Variable Oil':
            expand_order = pd.DataFrame()
            order_begin = r.order_begin
            order_end = r.order_end

            oco_order_no = 0
            # This line help to expand the order
            # create date first then fill the order_id otherwise order_id will be NaN
            expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
            expand_order['order_date'] = r.order_date
            expand_order['daily_quantity'] = r.daily_quantity
            expand_order['daily_bo_quantity'] = r.daily_bo_quantity
            expand_order['order_id'] = r.order_id

            expand_pvo_order = pd.concat([expand_order, expand_pvo_order], ignore_index=True)

            # end of loop
    return expand_pvo_order


def check_last_date(ftype='raw_data') -> str:
    # read pickle file
    latest_file = sorted(glob.iglob(f'Pickle/last_order*'), key=os.path.getctime, reverse=True)[0]
    print(f'[helpers.py @263] latest file for last_order:', latest_file)

    # # load pickle file
    data = pickle.load(open(latest_file, 'rb'))

    assert 'order_date' in data.columns, 'order_date column is missing!'

    # convert date column to pd.to_datetime
    data.order_date = pd.to_datetime(data.order_date)

    # get the last date
    return str(data.iloc[-1, :]['order_date'])[:10]


def preprocess_data(data, ftype='raw_data') -> pd.DataFrame:
    if ftype == 'raw_data':
        if not data.empty:
            data = (data.pipe(convert_datetime).pipe(join_market))
            data['market_short_code'] = data['market_short_code'].apply(lambda x: x[0])
        else:
            raise Exception('No data downloaded for the given date range!')

    # drop duplicates
    data.drop_duplicates(inplace=True)

    if ftype != 'raw_data':
        data.sort_values('order_date', inplace=True)

    data.reset_index(inplace=True)
    data.drop(columns='index', inplace=True)

    return data


def download_data(start_date, end_date, order_state, headers, market='"ASIA FO (PVO)"') -> pd.DataFrame:
    print(f'Start date: {start_date}, End date: {end_date}')

    url = 'https://api.platts.com/tradedata/v2/ewindowdata?filter='
    url = url + quote('market IN ({}) AND ORDER_DATE>="{}" AND ORDER_DATE<="{}" AND ORDER_STATE IN ({}) ' \
                      .format(market, start_date, end_date, order_state))
    temp_url = url + '&pagesize=1000&page=1'

    # find out page count
    print("start download ... ")
    time.sleep(3)

    response = requests.get(temp_url, headers=headers)
    data = response.json()
    page_count = data["metadata"]["total_pages"]
    print('Page count = ' + str(page_count))

    print('url = ' + temp_url)

    # sleep 2s otherwise API throws error
    time.sleep(3)

    # loop through pages
    symbol_data = []

    temp_url = url + '&pagesize=1000'
    print('url = ' + temp_url)
    print('Page count: %0.0f' % page_count)

    for page in range(1, page_count + 1):
        print("Pulling Page: ", page)
        # sleep 1s otherwise API throws error
        time.sleep(2)
        response = requests.get(temp_url + '&page={}'.format(page), headers=headers)
        data = response.json()
        symbol_data = symbol_data + data["results"]

    symbol_data = pd.DataFrame(symbol_data)
    print("Downnload completed!")

    return symbol_data


def convert_datetime(symbol_data):
    symbol_data['order_date'] = pd.to_datetime(symbol_data['order_date'], format="%Y-%m-%d",
                                               infer_datetime_format=True)  # save time by setting true
    symbol_data['order_begin'] = pd.to_datetime(symbol_data['order_begin'], format="%Y-%m-%d",
                                                infer_datetime_format=True)  # save time by setting true
    symbol_data['order_end'] = pd.to_datetime(symbol_data['order_end'], format="%Y-%m-%d",
                                              infer_datetime_format=True)  # save time by setting true
    symbol_data['deal_begin'] = pd.to_datetime(symbol_data['deal_begin'], format="%Y-%m-%d",
                                               infer_datetime_format=True)  # save time by setting true
    symbol_data['deal_end'] = pd.to_datetime(symbol_data['deal_end'], format="%Y-%m-%d",
                                             infer_datetime_format=True)  # save time by setting true
    symbol_data['order_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
                                               infer_datetime_format=True)
    symbol_data['update_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
                                                infer_datetime_format=True)
    # raw_order['order_year'] = pd.DatetimeIndex(raw_order['order_date']).year
    return symbol_data


def join_market(symbol_data):
    symbol_data['market'] = symbol_data['market'].apply(''.join)
    return symbol_data


def add_rank_group(df, group_size=3):
    group_col_name = 'rank_group'

    df.dropna(inplace=True)

    df['rank'] = df['rank'].astype(int)
    df[group_col_name] = df['rank'].apply(lambda x: f'_rgp_{(x - 1) // group_size + 1:02d}_sz_{group_size}')
    return df


# import pandas as pd
# import spgci as ci

# import os
# import requests
# import pickle
# import time
# import glob
# import datetime as dt
# from pathlib import Path
# import sqlite3

# import numpy as np
# from pandas.tseries.offsets import BDay

# from urllib.parse import quote
# from datetime import timedelta, datetime
# from pandas.tseries.offsets import *

# import sys
# print()


# def loggg(f):
#     def wrapper(*args, **kwargs):
#         tic = datetime.now()
#         result = f(*args, **kwargs)
#         toc = datetime.now()
#         print(f"{f.__name__} took {toc - tic} shape = {result.shape}")
#         return result

#     return wrapper


# def append_target(features, target, history=False):
#     # rename column
#     target.index.name = 'pricing_date'

#     # convert to datetime
#     target.index = pd.to_datetime(target.index)

#     features = features.join(target[['F380 M2/M3']], on='pricing_date')
#     features.dropna(inplace=True)

#     # reset index and drop duplicates
#     features = features.reset_index().drop_duplicates('pricing_date').set_index('pricing_date')

#     # concat with old values
#     if history:
#         # TO-DO: create target columns
#         old = pd.read_csv('data/data-2023-08-25.csv')
#         old.pricing_date = pd.to_datetime(old.pricing_date)
#         old.set_index('pricing_date', inplace=True)

#         return pd.concat([old, features], axis=0).reset_index().drop_duplicates('pricing_date')

#     return features


# def alternate_pivot(data):
#     # Get unique values for index and columns
#     index_vals = data['assessDate'].unique()
#     col_vals = data['symbol'].unique()

#     # Create an empty DataFrame to store the pivot result
#     result = pd.DataFrame(index=index_vals, columns=col_vals)

#     # Fill the result DataFrame using a loop
#     for index_value in index_vals:
#         for column_value in col_vals:
#             value = data[(data['assessDate'] == index_value) & (data['symbol'] == column_value)]['value'].mean()
#             result.at[index_value, column_value] = value

#     # Convert the result to numeric (optional, if not already numeric)
#     result = result.apply(pd.to_numeric)
#     return result


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
#     curves.to_csv('prices.csv')
#     return curves


# def download_curves_spgi(mdd, ric_dict, start_date='2017-01-01', end_date=dt.datetime.today()):
#     # download data
#     df = mdd.get_assessments_by_symbol_historical(
#         symbol=list(ric_dict.keys()),
#         bate=['c'],
#         assess_date_gte=start_date,
#         assess_date_lte=end_date,
#         paginate=True,
#     )

#     # pivot and rename columns
#     df = alternate_pivot(df)

#     if len(ric_dict) > 1:
#         df = df.rename(columns=ric_dict)

#     return df


# def format_data(pvo_expanded) -> pd.DataFrame:
#     # get company names list
#     columns = list(set(pvo_expanded['market_maker_mnemonic'].to_list()))
#     columns += list(set(pvo_expanded['rank_group'].to_list()))

#     # get dataframe with the values
#     company_names = pvo_expanded.pivot_table(columns=pvo_expanded['market_maker_mnemonic'],
#                                              index=pvo_expanded.pricing_date,
#                                              values=pvo_expanded[['daily_quantity']])

#     # get rgp groups
#     rgp_groups = pvo_expanded.pivot_table(columns=pvo_expanded['rank_group'],
#                                           index=pvo_expanded.pricing_date,
#                                           values=pvo_expanded[['daily_quantity']])

#     # combine datasets
#     combined = pd.concat([company_names, rgp_groups])
#     combined.columns = columns
#     combined.fillna(0, inplace=True)

#     return combined


# def append_pickle(new_data, path, file='asia'):
#     # get pickle file location
#     if file == 'asia':
#         name = 'Asia'
#         pickle_file = 'Pickle/raw_data-2017-01-01-2023-10-09.pickle'
#     elif file == 'last':
#         name = 'last_order'
#         pickle_file = 'Pickle/last_order-2017-01-01-2023-10-09.pickle'
#     else:
#         name = 'pvo_ranked'
#         pickle_file = 'Pickle/pvo_expanded-2017-01-01-2023-10-09.pickle'

#     # open pickle file
#     all_data = pickle.load(open(pickle_file, 'rb'))

#     if file == 'asia':
#         if 'market_short_code' in all_data.columns:
#             all_data['market_short_code'] = all_data['market_short_code'].apply(lambda x: x[0])

#     # standardize data
#     ftype = 'raw_data' if file == 'asia' else 'pvo'
#     all_data = preprocess_data(all_data, ftype)

#     # concat records
#     combined = pd.concat([all_data, new_data], axis=0)

#     # save to pickle
#     combined.to_pickle(f'{path}/' + name + str(dt.datetime.now())[:10])

#     return combined


# def get_last_order(symbol_data):
#     last_order = symbol_data[['order_time', 'order_type', 'order_state', 'update_time', 'market',
#                               'market_type', 'order_id', 'oco_order_id', 'deal_id', 'market_maker',
#                               'market_maker_mnemonic',
#                               'order_date', 'order_begin', 'order_end',
#                               'order_quantity', 'order_cancelled',
#                               'product', 'hub', 'strip', 'c1_basis_period', 'c1_basis_period_details', 'c1_price',
#                               'c1_price_basis', ]]
#     # sort and keep last
#     last_order = last_order.sort_values('order_time', ascending=True).drop_duplicates(['order_date', 'order_id'],
#                                                                                       keep='last')
#     # add new column with volume bid = positive offer = negative
#     last_order['bo_quantity'] = np.where((last_order['order_type'] == 'Offer'), last_order['order_quantity'] * -1,
#                                          last_order['order_quantity'])

#     last_order['oco_order_no'] = last_order['oco_order_id'].apply(
#         lambda x: 1 + round((len(x) + 1) / 9, 0) if x != None else 1)
#     last_order['pricing_days'] = np.busday_count(last_order.order_begin.values.astype('datetime64[D]'),
#                                                  last_order.order_end.values.astype('datetime64[D]'))
#     # #divide the total volume by number of oco contracts, if no oco, divide by 1
#     last_order['daily_quantity'] = last_order['order_quantity'] / (
#             last_order['pricing_days'] * last_order['oco_order_no'])
#     last_order['daily_bo_quantity'] = last_order['bo_quantity'] / (
#             last_order['pricing_days'] * last_order['oco_order_no'])
#     last_order['c1_price_basis'] = last_order['c1_price_basis'].str.rstrip('.')

#     return last_order



# def get_lo_vol_expanded(last_order):
#     expand_pvo_order = pd.DataFrame()

#     for index, r in last_order.iterrows():
#         # only expand phyiscal order
#         if r.market_type == 'Platts Variable Oil':
#             expand_order = pd.DataFrame()
#             order_begin = r.order_begin
#             order_end = r.order_end

#             oco_order_no = 0
#             # This line help to expand the order
#             # create date first then fill the order_id otherwise order_id will be NaN
#             expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
#             expand_order['order_date'] = r.order_date
#             expand_order['daily_quantity'] = r.daily_quantity
#             expand_order['daily_bo_quantity'] = r.daily_bo_quantity
#             expand_order['order_id'] = r.order_id

#             expand_pvo_order = pd.concat([expand_order, expand_pvo_order], ignore_index=True)

#             # end of loop
#     return expand_pvo_order


# def check_last_date(ftype='raw_data') -> str:
#     # read pickle file
#     latest_file = 'Pickle/last_order-2017-01-01-2023-10-09.pickle'

#     # # load pickle file
#     data = pickle.load(open(latest_file, 'rb'))

#     assert 'order_date' in data.columns, 'order_date column is missing!'

#     # convert date column to pd.to_datetime
#     data.order_date = pd.to_datetime(data.order_date)

#     # get the last date
#     return str(data.iloc[-1, :]['order_date'])[:10]


# def preprocess_data(data, ftype='raw_data') -> pd.DataFrame:
#     if ftype == 'raw_data':
#         if not data.empty:
#             data = (data.pipe(convert_datetime).pipe(join_market))
#             data['market_short_code'] = data['market_short_code'].apply(lambda x: x[0])
#         else:
#             raise Exception('No data downloaded for the given date range!')

#     # drop duplicates
#     data.drop_duplicates(inplace=True)

#     if ftype != 'raw_data':
#         data.sort_values('order_date', inplace=True)

#     data.reset_index(inplace=True)
#     data.drop(columns='index', inplace=True)

#     return data


# def download_data(start_date, end_date, order_state, headers, market='"ASIA FO (PVO)"') -> pd.DataFrame:
#     print(f'Start date: {start_date}, End date: {end_date}')

#     url = 'https://api.platts.com/tradedata/v2/ewindowdata?filter='
#     url = url + quote('market IN ({}) AND ORDER_DATE>="{}" AND ORDER_DATE<="{}" AND ORDER_STATE IN ({}) ' \
#                       .format(market, start_date, end_date, order_state))
#     temp_url = url + '&pagesize=1000&page=1'

#     # find out page count
#     print("start download ... ")
#     time.sleep(3)

#     response = requests.get(temp_url, headers=headers)
#     data = response.json()
#     page_count = data["metadata"]["total_pages"]
#     print('Page count = ' + str(page_count))

#     print('url = ' + temp_url)

#     # sleep 2s otherwise API throws error
#     time.sleep(3)

#     # loop through pages
#     symbol_data = []

#     temp_url = url + '&pagesize=1000'
#     print('url = ' + temp_url)
#     print('Page count: %0.0f' % page_count)

#     for page in range(1, page_count + 1):
#         print("Pulling Page: ", page)
#         # sleep 1s otherwise API throws error
#         time.sleep(2)
#         response = requests.get(temp_url + '&page={}'.format(page), headers=headers)
#         data = response.json()
#         symbol_data = symbol_data + data["results"]

#     symbol_data = pd.DataFrame(symbol_data)
#     print("Downnload completed!")

#     return symbol_data


# def convert_datetime(symbol_data):
#     symbol_data['order_date'] = pd.to_datetime(symbol_data['order_date'], format="%Y-%m-%d",
#                                                infer_datetime_format=True)  # save time by setting true
#     symbol_data['order_begin'] = pd.to_datetime(symbol_data['order_begin'], format="%Y-%m-%d",
#                                                 infer_datetime_format=True)  # save time by setting true
#     symbol_data['order_end'] = pd.to_datetime(symbol_data['order_end'], format="%Y-%m-%d",
#                                               infer_datetime_format=True)  # save time by setting true
#     symbol_data['deal_begin'] = pd.to_datetime(symbol_data['deal_begin'], format="%Y-%m-%d",
#                                                infer_datetime_format=True)  # save time by setting true
#     symbol_data['deal_end'] = pd.to_datetime(symbol_data['deal_end'], format="%Y-%m-%d",
#                                              infer_datetime_format=True)  # save time by setting true
#     symbol_data['order_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
#                                                infer_datetime_format=True)
#     symbol_data['update_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
#                                                 infer_datetime_format=True)
#     # raw_order['order_year'] = pd.DatetimeIndex(raw_order['order_date']).year
#     return symbol_data


# def join_market(symbol_data):
#     symbol_data['market'] = symbol_data['market'].apply(''.join)
#     return symbol_data


# def add_rank_group(df, group_size=3):
#     group_col_name = 'rank_group'

#     df.dropna(inplace=True)

#     df['rank'] = df['rank'].astype(int)
#     df[group_col_name] = df['rank'].apply(lambda x: f'_rgp_{(x - 1) // group_size + 1:02d}_sz_{group_size}')
#     return df


# # import pandas as pd
# # import spgci as ci

# # import os
# # import requests
# # import pickle
# # import time
# # import glob
# # import datetime as dt
# # from pathlib import Path
# # import sqlite3

# # import numpy as np
# # from pandas.tseries.offsets import BDay

# # from urllib.parse import quote
# # from datetime import timedelta, datetime
# # from pandas.tseries.offsets import *

# # import sys
# # print()


# # def loggg(f):
# #     def wrapper(*args, **kwargs):
# #         tic = datetime.now()
# #         result = f(*args, **kwargs)
# #         toc = datetime.now()
# #         print(f"{f.__name__} took {toc - tic} shape = {result.shape}")
# #         return result

# #     return wrapper


# # def append_target(features, target, history=False):
# #     # rename column
# #     target.index.name = 'pricing_date'

# #     # convert to datetime
# #     target.index = pd.to_datetime(target.index)

# #     features = features.join(target[['F380 M2/M3']], on='pricing_date')
# #     features.dropna(inplace=True)

# #     # reset index and drop duplicates
# #     features = features.reset_index().drop_duplicates('pricing_date').set_index('pricing_date')

# #     # concat with old values
# #     if history:
# #         # TO-DO: create target columns
# #         old = pd.read_csv('data/data-2023-08-25.csv')
# #         old.pricing_date = pd.to_datetime(old.pricing_date)
# #         old.set_index('pricing_date', inplace=True)

# #         return pd.concat([old, features], axis=0).reset_index().drop_duplicates('pricing_date')

# #     return features


# # def alternate_pivot(data):
# #     # Get unique values for index and columns
# #     index_vals = data['assessDate'].unique()
# #     col_vals = data['symbol'].unique()

# #     # Create an empty DataFrame to store the pivot result
# #     result = pd.DataFrame(index=index_vals, columns=col_vals)

# #     # Fill the result DataFrame using a loop
# #     for index_value in index_vals:
# #         for column_value in col_vals:
# #             value = data[(data['assessDate'] == index_value) & (data['symbol'] == column_value)]['value'].mean()
# #             result.at[index_value, column_value] = value

# #     # Convert the result to numeric (optional, if not already numeric)
# #     result = result.apply(pd.to_numeric)
# #     return result


# # def get_prices(mdd, start_date, end_date, target_col='F380 M2/M3'):
# #     ric_dict = {
# #         # Structure
# #         'AAWGS00': 'F380 M1/M2',
# #         'AAWGT00': 'F380 M2/M3',
# #         'AAWGU00': 'F380 M3/M4',

# #         # EW EU settlement
# #         'FQLSM01': 'EW F380 vs F35 M1',
# #         'FQLSM02': 'EW F380 vs F35 M2',
# #         'FQLSM03': 'EW F380 vs F35 M3',

# #         # Visco
# #         'AAVFL00': 'SG Visco M1',
# #         'AAVFM00': 'SG Visco M2',
# #         'AAVFN00': 'SG Visco M3',

# #         # Cracks
# #         'AAWHA00': 'F380 DB Cracks M1',
# #         'AAWHB00': 'F380 DB Cracks M2',

# #         'ABWDN00': 'ULSD NWE M1',
# #         'ABWDO00': 'ULSD NWE M2',
# #         'ABWDP00': 'ULSD NWE M3',
# #         'ABWDQ00': 'ULSD NWE M4',

# #         # Middle Distillate
# #         'AAXAL00': 'ULSD NWE vs ICEG M1',
# #         'AAXAM00': 'ULSD NWE vs ICEG M2'
# #     }

# #     curves = download_curves_spgi(mdd, ric_dict, start_date=start_date, end_date=end_date)
# #     curves.to_csv('prices.csv')
# #     return curves


# # def download_curves_spgi(mdd, ric_dict, start_date='2017-01-01', end_date=dt.datetime.today()):
# #     # download data
# #     df = mdd.get_assessments_by_symbol_historical(
# #         symbol=list(ric_dict.keys()),
# #         bate=['c'],
# #         assess_date_gte=start_date,
# #         assess_date_lte=end_date,
# #         paginate=True,
# #     )

# #     # pivot and rename columns
# #     df = alternate_pivot(df)

# #     if len(ric_dict) > 1:
# #         df = df.rename(columns=ric_dict)

# #     return df


# # def format_data(pvo_expanded) -> pd.DataFrame:
# #     # get company names list
# #     columns = list(set(pvo_expanded['market_maker_mnemonic'].to_list()))
# #     columns += list(set(pvo_expanded['rank_group'].to_list()))

# #     # get dataframe with the values
# #     company_names = pvo_expanded.pivot_table(columns=pvo_expanded['market_maker_mnemonic'],
# #                                              index=pvo_expanded.pricing_date,
# #                                              values=pvo_expanded[['daily_quantity']])

# #     # get rgp groups
# #     rgp_groups = pvo_expanded.pivot_table(columns=pvo_expanded['rank_group'],
# #                                           index=pvo_expanded.pricing_date,
# #                                           values=pvo_expanded[['daily_quantity']])

# #     # combine datasets
# #     combined = pd.concat([company_names, rgp_groups])
# #     combined.columns = columns
# #     combined.fillna(0, inplace=True)

# #     return combined


# # def append_pickle(new_data, file='asia'):
# #     # get pickle file location
# #     if file == 'asia':
# #         name = 'Asia'
# #         pickle_file = 'Pickle/raw_data-2017-01-01-2023-10-09'
# #     elif file == 'last':
# #         name = 'last_order'
# #         pickle_file = 'Pickle/last_order-2017-01-01-2023-10-09'
# #     else:
# #         name = 'pvo_ranked'
# #         pickle_file = 'Pickle/pvo_expanded-2017-01-01-2023-10-09'

# #     # open pickle file
# #     all_data = pickle.load(open(pickle_file, 'rb'))

# #     if file == 'asia':
# #         if 'market_short_code' in all_data.columns:
# #             all_data['market_short_code'] = all_data['market_short_code'].apply(lambda x: x[0])

# #     # standardize data
# #     ftype = 'raw_data' if file == 'asia' else 'pvo'
# #     all_data = preprocess_data(all_data, ftype)

# #     # concat records
# #     combined = pd.concat([all_data, new_data], axis=0)

# #     # save to pickle
# #     # combined.to_pickle('Pickle/' + name + str(dt.datetime.now())[:10])

# #     return combined


# # def get_lo_vol_expanded(last_order):
# #     expand_pvo_order = pd.DataFrame()

# #     for index, r in last_order.iterrows():
# #         # only expand phyiscal order
# #         if r.market_type == 'Platts Variable Oil':
# #             expand_order = pd.DataFrame()
# #             order_begin = r.order_begin
# #             order_end = r.order_end

# #             oco_order_no = 0
# #             # This line help to expand the order
# #             # create date first then fill the order_id otherwise order_id will be NaN
# #             expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
# #             expand_order['order_date'] = r.order_date
# #             expand_order['daily_quantity'] = r.daily_quantity
# #             expand_order['daily_bo_quantity'] = r.daily_bo_quantity
# #             expand_order['order_id'] = r.order_id

# #             # necessary columns for get_n_rolling function
# #             # expand_order['market'] = r['market']
# #             # expand_order['market_maker_mnemonic'] = r.market_maker_mnemonic
# #             # expand_order['order_quantity'] = r.order_quantity

# #             expand_pvo_order = pd.concat([expand_order, expand_pvo_order], ignore_index=True)

# #             # end of loop
# #     return expand_pvo_order


# # def add_original_data_to_db(db_name='airflow_pipeline.db'):
# #     # connect to the database
# #     con = sqlite3.connect(db_name)
    
# #     dfs, df_names = [], ['raw_data', 'last_order', 'lo_expanded']
    
# #     for name in df_names:
# #         if name != 'lo_expanded':
# #             data = pickle.load(open(f'Pickle/{name}-2017-01-01-2023-10-09', 'rb'))
# #         else:
# #             data = get_lo_vol_expanded(dfs[-1])
        
# #         if name == 'raw_order':
# #             data['market_short_code'] = data['market_short_code'].apply(lambda x: x[0])
        
# #         data = preprocess_data(data, 'raw' if 'raw' in name else 'pvo')
# #         data.sort_values('order_date').drop_duplicates()
        
# #         name = 'raw_order' if name == 'raw_data' else name
# #         data.to_sql(name, con=con, if_exists='replace', index=False)
# #         dfs.append(data)
    
# #     con.close()


# # def check_last_date(ftype='raw_data') -> str:
# #     import sys
# #     print(sys.path)
    
# #     # connect to the database
# #     con = sqlite3.connect('airflow_pipeline.db')
# #     cur = con.cursor()

# #     current_tables = cur.execute('SELECT * FROM sqlite_master').fetchall()
# #     print('Current tables:', current_tables)

# #     # getting last date for the data download
# #     if ftype == 'raw_data':
# #         if len(current_tables) == 0:        
# #             add_original_data_to_db()
        
# #         # get the last order
# #         last_order = pd.read_sql_query('SELECT * FROM last_order', con)
# #         last_order['order_date'] = pd.to_datetime(last_order['order_date'])
# #         return str(last_order['order_date'].max())[:10]

# #     # getting last date for ranking
# #     tables_names = [x[1] for x in current_tables]

# #     # get the last date if exists
# #     if 'daily_ranking' in tables_names:
# #         daily_ranking = pd.read_sql_query('SELECT * FROM daily_ranking', con)

# #         if len(daily_ranking) == 0 or 'order_date' not in daily_ranking.columns:
# #             return None

# #         daily_ranking['order_date'] = pd.to_datetime(daily_ranking['order_date'])
# #         return str(daily_ranking['order_date'].max())[:10]

# #     return None


# # def preprocess_data(data, ftype='raw_data') -> pd.DataFrame:
# #     if ftype == 'raw_data':
# #         if not data.empty:
# #             data = (data.pipe(convert_datetime).pipe(join_market))
# #             data['market_short_code'] = data['market_short_code'].apply(lambda x: x[0])
# #         else:
# #             raise Exception('No data downloaded for the given date range!')

# #     # drop duplicates
# #     data.drop_duplicates(inplace=True)

# #     if ftype != 'raw_data':
# #         data.sort_values('order_date', inplace=True)

# #     data.reset_index(inplace=True)
# #     data.drop(columns='index', inplace=True)

# #     return data


# # def download_data(start_date, end_date, order_state, headers, market='"ASIA FO (PVO)"') -> pd.DataFrame:
# #     print(f'Start date: {start_date}, End date: {end_date}')

# #     url = 'https://api.platts.com/tradedata/v2/ewindowdata?filter='
# #     url = url + quote('market IN ({}) AND ORDER_DATE>="{}" AND ORDER_DATE<="{}" AND ORDER_STATE IN ({}) ' \
# #                       .format(market, start_date, end_date, order_state))
# #     temp_url = url + '&pagesize=1000&page=1'

# #     # find out page count
# #     print("start download ... ")
# #     time.sleep(3)

# #     response = requests.get(temp_url, headers=headers)
# #     data = response.json()
# #     page_count = data["metadata"]["total_pages"]
# #     print('Page count = ' + str(page_count))

# #     print('url = ' + temp_url)

# #     # sleep 2s otherwise API throws error
# #     time.sleep(3)

# #     # loop through pages
# #     symbol_data = []

# #     temp_url = url + '&pagesize=1000'
# #     print('url = ' + temp_url)
# #     print('Page count: %0.0f' % page_count)

# #     for page in range(1, page_count + 1):
# #         print("Pulling Page: ", page)
# #         # sleep 1s otherwise API throws error
# #         time.sleep(2)
# #         response = requests.get(temp_url + '&page={}'.format(page), headers=headers)
# #         data = response.json()
# #         symbol_data = symbol_data + data["results"]

# #     symbol_data = pd.DataFrame(symbol_data)
# #     print("Downnload completed!")

# #     return symbol_data


# # def convert_datetime(symbol_data):
# #     symbol_data['order_date'] = pd.to_datetime(symbol_data['order_date'], format="%Y-%m-%d",
# #                                                infer_datetime_format=True)  # save time by setting true
# #     symbol_data['order_begin'] = pd.to_datetime(symbol_data['order_begin'], format="%Y-%m-%d",
# #                                                 infer_datetime_format=True)  # save time by setting true
# #     symbol_data['order_end'] = pd.to_datetime(symbol_data['order_end'], format="%Y-%m-%d",
# #                                               infer_datetime_format=True)  # save time by setting true
# #     symbol_data['deal_begin'] = pd.to_datetime(symbol_data['deal_begin'], format="%Y-%m-%d",
# #                                                infer_datetime_format=True)  # save time by setting true
# #     symbol_data['deal_end'] = pd.to_datetime(symbol_data['deal_end'], format="%Y-%m-%d",
# #                                              infer_datetime_format=True)  # save time by setting true
# #     symbol_data['order_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
# #                                                infer_datetime_format=True)
# #     symbol_data['update_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
# #                                                 infer_datetime_format=True)
# #     # raw_order['order_year'] = pd.DatetimeIndex(raw_order['order_date']).year
# #     return symbol_data


# # def join_market(symbol_data):
# #     symbol_data['market'] = symbol_data['market'].apply(''.join)
# #     return symbol_data


# # # def get_lo_expanded(last_order, history=False):
# # #     print(f'expand_pricing_dates: started ...')

# # #     pickle_path = 'Pickle/lo_expanded_07-31'

# # #     if os.path.exists(pickle_path) and history is False:
# # #         expand_pvo_order = pickle.load(open(pickle_path, 'rb'))
# # #     else:
# # #         expand_pvo_order = pd.DataFrame()

# # #     for index, r in last_order.iterrows():
# # #         if index in expand_pvo_order.index: continue

# # #         # only expand phyiscal order
# # #         if r.market_type == 'Platts Variable Oil':
# # #             expand_order = pd.DataFrame()
# # #             order_begin = r.order_begin
# # #             order_end = r.order_end

# # #             oco_order_no = 0
# # #             # This line help to expand the order
# # #             # create date first then fill the order_id otherwise order_id will be NaN
# # #             expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
# # #             # Fill in the order details
# # #             expand_order['market_maker'] = r.market_maker
# # #             expand_order['market'] = r['market']
# # #             expand_order['product'] = r['product']
# # #             expand_order['hub'] = r.hub

# # #             expand_order['order_type'] = r.order_type
# # #             expand_order['order_state'] = r.order_state

# # #             expand_order['market_maker_mnemonic'] = r.market_maker_mnemonic
# # #             expand_order['order_date'] = r.order_date

# # #             expand_order['oco_order_id'] = r.oco_order_id

# # #             expand_order['daily_quantity'] = r.daily_quantity
# # #             expand_order['daily_bo_quantity'] = r.daily_bo_quantity
# # #             expand_order['order_id'] = r.order_id
# # #             expand_order['c1_price_basis'] = r.c1_price_basis
# # #             expand_order['c1_price'] = r.c1_price
# # #             expand_order['c1_basis_period'] = r.c1_basis_period
# # #             expand_order['c1_basis_period_details'] = r.c1_basis_period_details
# # #             expand_order['order_quantity'] = r.order_quantity

# # #             # print('order id {} expand to {}'.format(r.order_id, len(expand_order['pricing_date'])))
# # #             # expand_pvo_order = expand_pvo_order.append(expand_order, ignore_index=True)
# # #             expand_pvo_order = pd.concat([expand_pvo_order, expand_order], ignore_index=True)

# # #             # end of loop

# # #     col_cat = ['market_maker', 'market', 'product', 'hub', 'order_type', 'order_state', 'market_maker_mnemonic',
# # #                'c1_price_basis', 'c1_basis_period', 'c1_basis_period_details']

# # #     expand_pvo_order[col_cat] = expand_pvo_order[col_cat].astype('category')

# # #     if not os.path.exists(pickle_path):
# # #         expand_pvo_order.to_pickle(pickle_path)

# # #     print(f'expand_pricing_dates: finished!')

# # #     return expand_pvo_order


# # def get_lo_vol_expanded(last_order):
# #     expand_pvo_order = pd.DataFrame()

# #     for index, r in last_order.iterrows():
# #         # only expand phyiscal order
# #         if r.market_type == 'Platts Variable Oil':
# #             expand_order = pd.DataFrame()
# #             order_begin = r.order_begin
# #             order_end = r.order_end

# #             oco_order_no = 0
# #             # This line help to expand the order
# #             # create date first then fill the order_id otherwise order_id will be NaN
# #             expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
# #             expand_order['order_date'] = r.order_date
# #             expand_order['daily_quantity'] = r.daily_quantity
# #             expand_order['daily_bo_quantity'] = r.daily_bo_quantity
# #             expand_order['order_id'] = r.order_id

# #             # necessary columns for get_n_rolling function
# #             # expand_order['market'] = r['market']
# #             # expand_order['market_maker_mnemonic'] = r.market_maker_mnemonic
# #             # expand_order['order_quantity'] = r.order_quantity

# #             expand_pvo_order = pd.concat([expand_order, expand_pvo_order], ignore_index=True)

# #             # end of loop
# #     return expand_pvo_order


# # def get_last_order(symbol_data):
# #     last_order = symbol_data[['order_time', 'order_type', 'order_state', 'update_time', 'market',
# #                               'market_type', 'order_id', 'oco_order_id', 'deal_id', 'market_maker',
# #                               'market_maker_mnemonic',
# #                               'order_date', 'order_begin', 'order_end',
# #                               'order_quantity', 'order_cancelled',
# #                               'product', 'hub', 'strip', 'c1_basis_period', 'c1_basis_period_details', 'c1_price',
# #                               'c1_price_basis', ]]
# #     # sort and keep last
# #     last_order = last_order.sort_values('order_time', ascending=True).drop_duplicates(['order_date', 'order_id'],
# #                                                                                       keep='last')
# #     # add new column with volume bid = positive offer = negative
# #     last_order['bo_quantity'] = np.where((last_order['order_type'] == 'Offer'), last_order['order_quantity'] * -1,
# #                                          last_order['order_quantity'])

# #     last_order['oco_order_no'] = last_order['oco_order_id'].apply(
# #         lambda x: 1 + round((len(x) + 1) / 9, 0) if x != None else 1)
# #     last_order['pricing_days'] = np.busday_count(last_order.order_begin.values.astype('datetime64[D]'),
# #                                                  last_order.order_end.values.astype('datetime64[D]'))
# #     # #divide the total volume by number of oco contracts, if no oco, divide by 1
# #     last_order['daily_quantity'] = last_order['order_quantity'] / (
# #             last_order['pricing_days'] * last_order['oco_order_no'])
# #     last_order['daily_bo_quantity'] = last_order['bo_quantity'] / (
# #             last_order['pricing_days'] * last_order['oco_order_no'])
# #     last_order['c1_price_basis'] = last_order['c1_price_basis'].str.rstrip('.')

# #     return last_order


# # # New Top Market Maker N Day back
# # def get_n_period_rolling_rank(df, n=90, ):
# #     """
# #     Calculates the rolling N-day ranking of the sales volume of a company in a given dataset.

# #     Parameters:
# #     df (pandas.DataFrame): The dataset containing company and sales volume information.
# #     n (int): The number of days to consider for the rolling ranking.

# #     Returns:
# #     pandas.DataFrame: The original DataFrame with an added column for the rolling N-day ranking of the sales volume.
# #     """

# #     company = 'market_maker_mnemonic'
# #     quantity = 'order_quantity'
# #     date = 'order_date'
# #     market = 'market'

# #     # create a day | lvl 1 - market, lvl2 - company.... company matrix for cum sum
# #     # this is key to align dates
# #     pivot = df.pivot_table(index=date, columns=[market, company], values=quantity, aggfunc=np.sum).fillna(0)

# #     # pivot = alternate_pivot_rolling_rank(df, date, market, company, quantity)

# #     # rolling sum
# #     pivot.rolling(n).sum().dropna()
# #     rank = pivot.rolling(n).sum().dropna().melt(ignore_index=False).reset_index()
# #     rank['rank'] = rank.groupby([market, date])['value'].rank(method='min', ascending=False)
# #     rank = rank.sort_values(by=['order_date', 'rank'], ascending=[True, True])
# #     return rank


# # def add_ranks(pvo_ranked, last_order, periods=30):
# #     ranks = get_n_period_rolling_rank(df=last_order, n=periods)

# #     ranks = ranks.drop('value', axis=1)

# #     # Join back: join ranking back into the expanded
# #     pvo_ranked = pd.merge(pvo_ranked, ranks, on=['order_date', 'market', 'market_maker_mnemonic'])

# #     if 'rank_y' in pvo_ranked.columns.to_list():
# #         pvo_ranked.drop(columns='rank_y', inplace=True)
# #         pvo_ranked.rename(columns={'rank_x': 'rank'}, inplace=True)

# #     print('pvo ranked columns:', pvo_ranked.columns)

# #     return pvo_ranked


# # def add_rank_group(df, group_size=3):
# #     group_col_name = 'rank_group'

# #     df.dropna(inplace=True)

# #     df['rank'] = df['rank'].astype(int)
# #     df[group_col_name] = df['rank'].apply(lambda x: f'_rgp_{(x - 1) // group_size + 1:02d}_sz_{group_size}')
# #     return df


# # def get_last_file(keyword) -> str:
# #     # look for last file matching description
# #     latest_file = sorted(glob.iglob('Pickle/*'), key=os.path.getctime, reverse=True)
# #     return [x for x in latest_file if keyword in x][0]


# # # import pandas as pd
# # # import spgci as ci
# # #
# # # import os
# # # import requests
# # # import pickle
# # # import time
# # # import glob
# # # import datetime as dt
# # # from pathlib import Path
# # # import sqlite3
# # #
# # # import numpy as np
# # # from pandas.tseries.offsets import BDay
# # #
# # # from urllib.parse import quote
# # # from datetime import timedelta, datetime
# # # from pandas.tseries.offsets import *
# # #
# # #
# # # def get_last_file(keyword) -> str:
# # #     # look for last file matching description
# # #     latest_file = sorted(glob.iglob('../../Pickle/*'), key=os.path.getctime, reverse=True)
# # #     return [x for x in latest_file if keyword in x][0]
# # #
# # #
# # # def get_headers():
# # #     API_TOKEN = 'hsZhxtLPXCvgrIagPFpq'
# # #     HEADERS = {'appkey': API_TOKEN, 'Content-Type': 'application/json'}
# # #
# # #     # set credentials
# # #     ci.set_credentials("simon.xin@mlp.com", "Godbless!001", "HJrqeVqsRsWEYqmxpGBU")
# # #     mdd = ci.MarketData()
# # #
# # #     return HEADERS, mdd
# # #
# # #
# # # def download_data(start_date, end_date, order_state, headers, market='"ASIA FO (PVO)"') -> pd.DataFrame:
# # #     print(f'Start date: {start_date}, End date: {end_date}')
# # #
# # #     url = 'https://api.platts.com/tradedata/v2/ewindowdata?filter='
# # #     url = url + quote('market IN ({}) AND ORDER_DATE>="{}" AND ORDER_DATE<="{}" AND ORDER_STATE IN ({}) ' \
# # #                       .format(market, start_date, end_date, order_state))
# # #     temp_url = url + '&pagesize=1000&page=1'
# # #
# # #     # find out page count
# # #     print("start download ... ")
# # #     time.sleep(3)
# # #
# # #     response = requests.get(temp_url, headers=headers)
# # #     data = response.json()
# # #     page_count = data["metadata"]["total_pages"]
# # #     print('Page count = ' + str(page_count))
# # #
# # #     print('url = ' + temp_url)
# # #
# # #     # sleep 2s otherwise API throws error
# # #     time.sleep(3)
# # #
# # #     # loop through pages
# # #     symbol_data = []
# # #
# # #     temp_url = url + '&pagesize=1000'
# # #     print('url = ' + temp_url)
# # #     print('Page count: %0.0f' % page_count)
# # #
# # #     for page in range(1, page_count + 1):
# # #         print("Pulling Page: ", page)
# # #         # sleep 1s otherwise API throws error
# # #         time.sleep(2)
# # #         response = requests.get(temp_url + '&page={}'.format(page), headers=headers)
# # #         data = response.json()
# # #         symbol_data = symbol_data + data["results"]
# # #
# # #     symbol_data = pd.DataFrame(symbol_data)
# # #     print("Downnload completed!")
# # #
# # #     return symbol_data
# # #
# # #
# # # def preprocess_data(data, ftype='raw_data') -> pd.DataFrame:
# # #     if ftype == 'raw_data':
# # #         if not data.empty:
# # #             data = (data.pipe(convert_datetime).pipe(join_market))
# # #             data['market_short_code'] = data['market_short_code'].apply(lambda x: x[0])
# # #         else:
# # #             raise Exception('No data downloaded for the given date range!')
# # #
# # #     # drop duplicates
# # #     data.drop_duplicates(inplace=True)
# # #
# # #     if ftype != 'raw_data':
# # #         data.sort_values('order_date', inplace=True)
# # #
# # #     data.reset_index(inplace=True)
# # #     data.drop(columns='index', inplace=True)
# # #
# # #     return data
# # #
# # #
# # # def convert_datetime(symbol_data):
# # #     symbol_data['order_date'] = pd.to_datetime(symbol_data['order_date'], format="%Y-%m-%d",
# # #                                                infer_datetime_format=True)  # save time by setting true
# # #     symbol_data['order_begin'] = pd.to_datetime(symbol_data['order_begin'], format="%Y-%m-%d",
# # #                                                 infer_datetime_format=True)  # save time by setting true
# # #     symbol_data['order_end'] = pd.to_datetime(symbol_data['order_end'], format="%Y-%m-%d",
# # #                                               infer_datetime_format=True)  # save time by setting true
# # #     symbol_data['deal_begin'] = pd.to_datetime(symbol_data['deal_begin'], format="%Y-%m-%d",
# # #                                                infer_datetime_format=True)  # save time by setting true
# # #     symbol_data['deal_end'] = pd.to_datetime(symbol_data['deal_end'], format="%Y-%m-%d",
# # #                                              infer_datetime_format=True)  # save time by setting true
# # #     symbol_data['order_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
# # #                                                infer_datetime_format=True)
# # #     symbol_data['update_time'] = pd.to_datetime(symbol_data['order_time'], format="%Y-%m-%dT%H:%M:%S.%f",
# # #                                                 infer_datetime_format=True)
# # #     # raw_order['order_year'] = pd.DatetimeIndex(raw_order['order_date']).year
# # #     return symbol_data
# # #
# # #
# # # def join_market(symbol_data):
# # #     symbol_data['market'] = symbol_data['market'].apply(''.join)
# # #     return symbol_data
# # #
# # #
# # # def check_last_date(option='pickle') -> str:
# # #     if option == 'pickle':
# # #         # read pickle file
# # #         # latest_file = sorted(glob.iglob('Pickle/last_order_FO_2023-07-31-132952'), key=os.path.getctime, reverse=True)
# # #         # latest_file = [x for x in latest_file if 'pvo_ranked' in x][0]
# # #         latest_file = '../../Pickle/last_order_FO_2023-07-31-132952'
# # #
# # #         # # load pickle file
# # #         data = pickle.load(open(latest_file, 'rb'))
# # #
# # #         assert 'order_date' in data.columns, 'order_date column is missing!'
# # #
# # #         # convert date column to pd.to_datetime
# # #         data.order_date = pd.to_datetime(data.order_date)
# # #
# # #         # get the last date
# # #         return str(data.iloc[-1, :]['order_date'])[:10]
# # #
# # #
# # # def get_lo_vol_expanded(last_order):
# # #     expand_pvo_order = pd.DataFrame()
# # #
# # #     for index, r in last_order.iterrows():
# # #         # only expand phyiscal order
# # #         if r.market_type == 'Platts Variable Oil':
# # #             expand_order = pd.DataFrame()
# # #             order_begin = r.order_begin
# # #             order_end = r.order_end
# # #
# # #             oco_order_no = 0
# # #             # This line help to expand the order
# # #             # create date first then fill the order_id otherwise order_id will be NaN
# # #             expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
# # #             expand_order['order_date'] = r.order_date
# # #             expand_order['daily_quantity'] = r.daily_quantity
# # #             expand_order['daily_bo_quantity'] = r.daily_bo_quantity
# # #             expand_order['order_id'] = r.order_id
# # #
# # #             # necessary columns for get_n_rolling function
# # #             # expand_order['market'] = r['market']
# # #             # expand_order['market_maker_mnemonic'] = r.market_maker_mnemonic
# # #             # expand_order['order_quantity'] = r.order_quantity
# # #
# # #             expand_pvo_order = pd.concat([expand_order, expand_pvo_order], ignore_index=True)
# # #
# # #             # end of loop
# # #     return expand_pvo_order
# # #
# # #
# # # def get_lo_expanded(last_order):
# # #     print(f'expand_pricing_dates: started ...')
# # #
# # #     expand_pvo_order = pd.DataFrame()
# # #
# # #     for index, r in last_order.iterrows():
# # #         # only expand phyiscal order
# # #         if r.market_type == 'Platts Variable Oil':
# # #             expand_order = pd.DataFrame()
# # #             order_begin = r.order_begin
# # #             order_end = r.order_end
# # #
# # #             oco_order_no = 0
# # #             # This line help to expand the order
# # #             # create date first then fill the order_id otherwise order_id will be NaN
# # #             expand_order['pricing_date'] = pd.bdate_range(start=order_begin, end=order_end, freq="B")
# # #             # Fill in the order details
# # #             expand_order['market_maker'] = r.market_maker
# # #             expand_order['market'] = r['market']
# # #             expand_order['product'] = r['product']
# # #             expand_order['hub'] = r.hub
# # #
# # #             expand_order['order_type'] = r.order_type
# # #             expand_order['order_state'] = r.order_state
# # #
# # #             expand_order['market_maker_mnemonic'] = r.market_maker_mnemonic
# # #             expand_order['order_date'] = r.order_date
# # #
# # #             expand_order['oco_order_id'] = r.oco_order_id
# # #
# # #             expand_order['daily_quantity'] = r.daily_quantity
# # #             expand_order['daily_bo_quantity'] = r.daily_bo_quantity
# # #             expand_order['order_id'] = r.order_id
# # #             expand_order['c1_price_basis'] = r.c1_price_basis
# # #             expand_order['c1_price'] = r.c1_price
# # #             expand_order['c1_basis_period'] = r.c1_basis_period
# # #             expand_order['c1_basis_period_details'] = r.c1_basis_period_details
# # #
# # #             # print('order id {} expand to {}'.format(r.order_id, len(expand_order['pricing_date'])))
# # #             # expand_pvo_order = expand_pvo_order.append(expand_order, ignore_index=True)
# # #             expand_pvo_order = pd.concat([expand_pvo_order, expand_order], ignore_index=True)
# # #
# # #             # end of loop
# # #
# # #     col_cat = ['market_maker', 'market', 'product', 'hub', 'order_type', 'order_state', 'market_maker_mnemonic',
# # #                'c1_price_basis', 'c1_basis_period', 'c1_basis_period_details']
# # #
# # #     expand_pvo_order[col_cat] = expand_pvo_order[col_cat].astype('category')
# # #
# # #     print(f'expand_pricing_dates: finished!')
# # #
# # #     return expand_pvo_order
# # #
# # #
# # # def download_curves_spgi(mdd, ric_dict, start_date='2017-01-01', end_date=dt.datetime.today()):
# # #     # download data
# # #     df = mdd.get_assessments_by_symbol_historical(
# # #         symbol=list(ric_dict.keys()),
# # #         bate=['c'],
# # #         assess_date_gte=start_date,
# # #         assess_date_lte=end_date,
# # #         paginate=True,
# # #     )
# # #
# # #     # pivot and rename columns
# # #     df = df.pivot_table(index='assessDate', columns='symbol', values='value', aggfunc='mean')
# # #     if len(ric_dict) > 1:
# # #         df = df.rename(columns=ric_dict)
# # #
# # #     return df
# # #
# # #
# # # def start_fill_rank(last_order, lag, frequency='M', period=180):
# # #     # get ranks
# # #     daily_rank = get_n_period_rolling_rank(last_order, n=period)
# # #
# # #     # convert to monthly rank
# # #     daily_rank.set_index('order_date', inplace=True)
# # #     daily_rank.to_csv('daily_rank.csv')
# # #
# # #     # if frequency == 'M':
# # #     monthly_rank = daily_rank.resample('M').last().shift(lag)
# # #     monthly_rank.to_csv('monthly_rank.csv')
# # #
# # #
# # #
# # #
# # #
# # # def get_n_period_rolling_rank(df, n=180):
# # #     """
# # #     Calculates the rolling N-day ranking of the sales volume of a company in a given dataset.
# # #
# # #     Parameters:
# # #     df (pandas.DataFrame): The dataset containing company and sales volume information.
# # #     n (int): The number of days to consider for the rolling ranking.
# # #
# # #     Returns:
# # #     pandas.DataFrame: The original DataFrame with an added column for the rolling N-day ranking of the sales volume.
# # #     """
# # #
# # #     company = 'market_maker_mnemonic'
# # #     quantity = 'order_quantity'
# # #     date = 'order_date'
# # #     market = 'market'
# # #     # effective = ''
# # #
# # #     # create a day | lvl 1 - market, lvl2 - company.... company matrix for cum sum
# # #     # this is key to align dates
# # #     pivot = df.pivot_table(index=date, columns=[market, company], values=quantity, aggfunc=np.sum).fillna(0)
# # #     # rolling sum
# # #     pivot.rolling(n).sum().dropna()
# # #     rank = pivot.rolling(n).sum().dropna().melt(ignore_index=False).reset_index()
# # #     rank['rank'] = rank.groupby([market, date])['value'].rank(method='min', ascending=False)
# # #     rank = rank.sort_values(by=['order_date', 'rank'], ascending=[True, True])
# # #     return rank
