import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from data import data
from model import model

import billiard as multiprocessing
import matplotlib.pyplot as plt
import tsfresh
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
import pickle
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay

import json
import sys

sys.path.append('../')

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

CRITICAL = 'CRITICAL'


def _check_data(today, log, ti) -> str:    
    log_path, save_path, target = ti['log_path'], ti['save_path'], ti['target_col']
    
    # check if today's data was already created
    df_path = f'data/data_{today}.csv'
    
    if not os.path.exists(df_path):
        log(log_path, f'[check_data]: Creating dataset for {today} date.')

        data.start_download(today, save_path, df_path, target_col=target)
    else:
        # load dataset
        df = pd.read_csv(df_path)
        df.pricing_date = pd.to_datetime(df.pricing_date)
        
        # check if data exists
        download = check_dates(df, ti)
        
        if not download:
            log(log_path, f'[check_data]: Creating dataset for {today} date.')
            data.start_download(today, save_path, df_path, target_col=target)
    
        log(log_path, f'[check_data]: Loading existing data for {today} date.')
        
    return df_path        


def check_dates(data, ti):
    test_start, test_end = ti['test_start'], ti['test_end']
    
    if data.pricing_date.max() < pd.to_datetime(test_end):
        return False
    
    return True


def non_consecutive_days(df, column='F380 M2/M3'):
    df = pd.concat([
    df, 
        (
            df[column].isnull().astype(int)
            .groupby(df[column].notnull().astype(int).cumsum())
            .cumsum().to_frame('consec_count')
        )
    ],
    axis=1)
    
    return str(df[df['consec_count'] == 1].index.max())[:10]


def _prepare_data(today, log, data_path, model_type, ti):
    log_path, save_path, target = ti['log_path'], ti['save_path'], ti['target_col']
    train_start, train_end, test_start, test_end = ti['train_start'], ti['train_end'], ti['test_start'], ti['test_end']
    
    results_folder, variance = save_path, 0.2
    
    # load data
    data = pd.read_csv(data_path)
    data.pricing_date = pd.to_datetime(data.pricing_date)
    data.set_index('pricing_date', inplace=True)
    print('data @83:', data)
    
    print('non consecutive:', non_consecutive_days(data[[target]]))
    
    # slice the dataset
    # data = data[data.index < non_consecutive_days(data[[target]])]
    data.interpolate(method='linear', inplace=True)
    
    data = data[(data.index >= train_start) & (data.index <= test_end)]
    data[target] = data[target].fillna(0)
    
    # rolling MA(5)
    # data[target] = data[target].rolling(5).mean()
    # data = data.iloc[5:, :]
    
    log(log_path, f'Slicing the data to start from {train_start} and end with {test_end} dates.')

    print('data @93:\n', data)

    # drop missing columns
    old_shape = data.shape[0]
    data.dropna(inplace=True)
    log(log_path, f'Dropped missing columns. Number of missing columns: {data.shape[0] - old_shape}.')

    # separate target and feature columns
    target_feature = TargetFeature(log_path, log, results_path=results_folder,
                                   data=data, test_start=test_start, test_end=test_end)
    target_feature()
    log(log_path, 'Created TargetFeature class.')
    
    # add time-related features if not present
    data = add_time_features(data, log=log, log_path=log_path)
    
    if model_type == 'xgboost':
        # drop target column
        data.drop(columns=target, inplace=True)
    
        print('data @109:\n', data)

        # get columns with variance bigger than the specified variance (default=0.2)
        columns = drop_low_variance(data, test_start, log, log_path, test_end, var=variance)
        # columns = list(data.columns)
        print('columns:', columns)
        log(log_path, f'Selected {len(columns)} with variance={variance}: {columns}.')

        # normalize data
        train, test = data.loc[:test_start, :][columns], data.loc[test_start:, :][columns]
        feature_scaler = MinMaxScaler().fit(train)

        train = pd.DataFrame(feature_scaler.transform(train), index=train.index, columns=train.columns)
        test = pd.DataFrame(feature_scaler.transform(test), index=test.index, columns=test.columns)

        # final log
        log(log_path, f'Task 2: Completed. Returning data, columns, and TargetFeature class instance.')

        # save current state in a pickle
        pickle_loc = f'{results_folder}/pickles'
        pickle_file = f'{pickle_loc}/results_variance={variance}'
        
        print('PICKLE FILE CREATED:', pickle_file)

        # create a folder to store results
        if not os.path.exists(pickle_loc): os.mkdir(pickle_loc)

        # store results
        with open(pickle_file, 'wb') as fout:
            pickle.dump({'data': data, 'columns': columns, 'target_feature': target_feature,
                        'test_end': test_end, 'train': train, 'test': test, 'feature_scaler': feature_scaler},
                        fout, pickle.HIGHEST_PROTOCOL)

        return pickle_file
    
    # else:
        # features, targets = data.drop(columns=target), data[target]
        # features.reset_index(inplace=True)
        
        # kind_fc_params = json.load(open('data/kind_fc_params.json'))
        
        # extracted = tsfresh.extract_features(features, column_sort='pricing_date', column_id='pricing_date', kind_to_fc_parameters=kind_fc_params, n_jobs=4)
        # # impute(extracted)
        # # features_filtered = tsfresh.select_features(extracted, targets)
    
        
        # pickle_loc = f'{save_path}/pickles'
        # pickle_file = f'{pickle_loc}/tsfresh_data'

        # # create a folder to store results
        # if not os.path.exists(pickle_loc): os.mkdir(pickle_loc)

        # # store results
        # with open(pickle_file, 'wb') as fout:
        #     pickle.dump({'features_filtered': extracted, 'y': targets}, fout, pickle.HIGHEST_PROTOCOL)

        # return pickle_file



def drop_low_variance(data, test_start, log, log_path, test_end, var=0.2):
    # log information
    log(log_path, f'Task 2: Selecting and dropping features with low variance (var < {var})...')

    # remove rows after test start
    data = data[(data.index > '2017-03-21') & (data.index < test_start)]

    # check the length of prediction dates
    if len(data[(data.index >= test_start) & (data.index < test_end)]) < 100:
        return data.columns[data.var() > var]

    return data[['_rgp_01_sz_3', '_rgp_02_sz_3', '_rgp_03_sz_3', '_rgp_04_sz_3', '_rgp_05_sz_3',
                 'BPSG', 'COASTAL', 'GUNVORSG', 'HL', 'MERCURIASG', 'P66SG', 'PETROCHINA',
                 'SIETCO', 'TOTALSG', 'TRAFI', 'VITOLSG']].columns



def _train_ridge(today, z_values=[10**4,  10**5, 10**6, 10**7, 10**8], ti=None):
    save_path = ti['save_path'] 
    test_start, test_end = ti['test_start'], ti['test_end']
    
    # ========================================= DATA PREPARATION PIPELINE =========================================
    
    df = pd.read_csv(f'data/data_{today}.csv')
    
    assert isinstance(df, pd.DataFrame) and len(df) > 0, 'The dataset does not exist! [@main.py, L:220]'
    
    # convert dates
    df.pricing_date = pd.to_datetime(df.pricing_date)
    df.set_index('pricing_date', inplace=True)
    
    # slice from train_start until test_end
    df = df[(df.index >= ti['train_start']) & (df.index < ti['test_end'])]
    df.interpolate(method='linear', inplace=True)
    df.fillna(0, inplace=True)
    
    features, target = df.drop(columns=ti['target_col']), df[ti['target_col']]
    features.reset_index(inplace=True)
    
    # target.fillna(0, inplace=True)
    # features.interpolate(inplace=True)
    
    # interpolate values (interpolation does not 'guess' the future values, they still end up being NaN)
    # features.interpolate(inplace=True)
    # target.interpolate(inplace=True)
    # target.fillna(inplace=True)     # otherwise breaks the ridge implementation
    
    # extract relevant features that were already ran (saves a lot of time)
    kind_fc_params = json.load(open('data/kind_fc_params.json'))
    features_filtered = tsfresh.extract_features(features, column_sort='pricing_date', column_id='pricing_date', kind_to_fc_parameters=kind_fc_params, n_jobs=4)
    impute(features_filtered)
    
    # save tsfresh data for future reference
    pickle_loc = f'{save_path}/pickles'
    pickle_file = f'{pickle_loc}/tsfresh_data'

    # create a folder to store results
    if not os.path.exists(pickle_loc): os.mkdir(pickle_loc)

    # store results
    with open(pickle_file, 'wb') as fout:
        pickle.dump({'features_filtered': features_filtered, 'y': target}, fout, pickle.HIGHEST_PROTOCOL)
    
    
    def train(z, features, target, test_start):
        y = target
        
        results_df = pd.DataFrame(columns=['pricing_date', 'preds', 'target', 'forecast_date'])

        t_values = y[y.index >= test_start].index

        for t in t_values:
            test_start = str(t)[:10]
            
            R = y[y.index < test_start]                                         # y_train
            R_s = y[y.index >= test_start]                                      # y_test
            S = features[features.index < test_start]                           # x_train
            S_t = features[features.index >= test_start]                        # x_test

            beta = Ridge(alpha=z, solver='svd', fit_intercept=False).fit(S, R).coef_
            
            forecast = S_t @ beta

            results_df = pd.concat([results_df, pd.DataFrame({
                'pricing_date': [test_start],
                'preds': [forecast.values[0]],
                'target': [R_s.values[0]],
            })], ignore_index=True)
                        
            results_df.to_excel(os.path.join(save_path, f'tsfresh_virtue_z={z}.xlsx'), sheet_name=f'z={z}')
            model.show_plot(results_df, os.path.join(save_path, f'tsfresh_virtue_z={z}.png'))
            
    for z in z_values:
        train(z, features_filtered.copy(), target.copy(), test_start=test_start)
        
    
    # train, test = features_filtered[features_filtered.index < test_start], features_filtered[(features_filtered.index >= test_start) & (features_filtered.index <= test_end)]
    
    # results_df = pd.DataFrame(columns=['pricing_date', 'preds', 'target', 'forecast_date'])
    
    # # scaler = MinMaxScaler().fit(train)
    # # y_scaler = MinMaxScaler().fit(pd.DataFrame(y[y.index < test.index.min()]))
    
    # for z_val in z:
    #     for date in test.index.tolist():
    #         date = str(date)[:10]
            
    #         full_history = features_filtered.copy()
    #         X_train = full_history[full_history.index < date]
    #         y_train = y[y.index < date]
            
    #         X_test, y_test = full_history[full_history.index >= date], y[y.index >= date]
            
    #         beta = Ridge(alpha=z_val, solver='svd', fit_intercept=False).fit(X_train, y_train).coef_
            
    #         forecast = X_test @ beta
            
    #         results_df = pd.concat([results_df, pd.DataFrame({
    #             'pricing_date': [date],
    #             'preds': forecast[0],
    #             'target': y_test,
    #         })], ignore_index=True)
                        
    #     results_df.to_csv(os.path.join(save_path, f'tsfresh_virtue_z={z_val}.csv'))
    #     model.show_plot(results_df, os.path.join(save_path, f'tsfresh_virtue_z={z_val}.png'))



def _pretrain_model(today, log, ti):
    log_path, save_path, target = ti['log_path'], ti['save_path'], ti['target_col']
    train_start, train_end, test_start, test_end = ti['train_start'], ti['train_end'], ti['test_start'], ti['test_end']
    
    log(log_path, 'Initiating Task 3: pretraining model...')
    log(log_path, 'Task 3: getting values from xcom')
    
    data_file = 'data/data_' + today + '.csv'
    results_folder, variance = save_path, 0.2
    
    print('Data file:',data_file)

    if os.path.exists(data_file):
        results = pd.read_csv(data_file)
    else:
        try:
            results = pickle.load(open(data_file, 'rb'))
        except TypeError:
            log(log_path, 'No data was pushed!', 'CRITICAL')
            raise TypeError('No data was created!')
    
    print('results:', results)

    # load pickle results
    data_file = f'{results_folder}/pickles/results_variance={variance}'

    if os.path.exists(data_file):
        results = pickle.load(open(f'{results_folder}/pickles/results_variance={variance}', 'rb'))

    if 'data' not in results.keys():
        log(log_path, 'Task 3: MISSING `data` KEY IN PICKLE FILE', CRITICAL)
        raise KeyError

    if 'columns' not in results.keys():
        log(log_path, 'Task 3: MISSING `columns` KEY IN PICKLE FILE', CRITICAL)
        raise KeyError

    if 'target_feature' not in results.keys():
        log(log_path, 'Task 3: MISSING `target_feature` KEY IN PICKLE FILE', CRITICAL)
        raise KeyError

    # unpack pickle file and get files
    data, columns, target_feature, test_end = results['data'], results['columns'], results['target_feature'], results['test_end']

    # create model name
    log(log_path, 'Task 3: Creating a model name...')
    model_name = create_model_name(results_folder, variance, len(columns), train_end)
    log(log_path, f'Task 3: Created a model name - {model_name}.')

    # pretrain model
    log(log_path, 'Task 3: Deciding whether to pretrain the model from scratch or load existing model.')
    
    print('ti keys:', ti.keys())

    if 'test_end' not in ti.keys():
        xgb_model = model.pretrain_model(model_name, target_feature, results_folder=results_folder, variance=variance, ti=ti)
    else:
        xgb_model = model.pretrain_model(model_name, target_feature, results_folder=results_folder, variance=variance, ti=ti)

    # save current state in a pickle
    pickle_loc = f'{results_folder}/pickles'
    pickle_file = f'{pickle_loc}/model_variance={variance}'

    with open(pickle_file, 'wb') as fout:
        pickle.dump({'model': xgb_model}, fout, pickle.HIGHEST_PROTOCOL)

    return pickle_file


def _get_predictions(today, log, ti):
    log_path, save_path, target = ti['log_path'], ti['save_path'], ti['target_col']
    train_start, train_end, test_start, test_end = ti['train_start'], ti['train_end'], ti['test_start'], ti['test_end']
    
    results_folder, variance = save_path, 0.2
    
    log(log_path, 'Initiating Task 4: getting predictions...')

    # load pickle results
    results = pickle.load(open(f'{results_folder}/pickles/results_variance={variance}', 'rb'))
    log(log_path, 'Task 4: loaded results from pickle.')

    # get values from pickle
    data, columns, target_feature = results['data'], results['columns'], results['target_feature']

    log(log_path, 'Task 4: loading pretrained model...')
    xgb_model = pickle.load(open(f'{results_folder}/pickles/model_variance={variance}', 'rb'))['model']
    log(log_path, 'Task 4: loaded model')

    log(log_path, 'Task 4: getting predictions...')

    results = model.get_predictions(data, columns, target_feature, xgb_model, results_folder, variance,
                                    train_start, train_end, test_start, test_end, log, log_path)

    # save current state in a pickle
    pickle_loc = f'{results_folder}/pickles'
    pickle_file = f'{pickle_loc}/predictions'

    with open(pickle_file, 'wb') as fout:
        pickle.dump(results, fout, pickle.HIGHEST_PROTOCOL)

    return pickle_file

    
def create_model_name(results_folder, variance, col_len, train_end):
    folder_path = f'experiments'

    # create global folders
    model_path = os.path.join(folder_path, 'models')

    # create folder if it doesn't exist
    if not os.path.exists(folder_path): os.mkdir(folder_path)
    if not os.path.exists(model_path): os.mkdir(model_path)

    model_date = train_end[:4]

    model_name = f'train-until-{model_date}_variance={variance}_col_length={col_len}.model'
    return os.path.join(model_path, model_name)


def add_time_features(data, log, log_path):
    log(log_path, 'Task 2: Adding time-related features...')

    if 'year_sin' not in data.columns:
        data['year_sin'] = np.sin(data.index.year / data.index.year.max() * 2 * np.pi)
        log(log_path, 'Added `year_sin` feature.')

    if 'month_sin' not in data.columns:
        data['month_sin'] = np.sin(data.index.month / data.index.month.max() * 2 * np.pi)
        log(log_path, 'Added `month_sin` feature.')

    if 'day_sin' not in data.columns:
        data['day_sin'] = np.sin(data.index.day / data.index.day.max() * 2 * np.pi)
        log(log_path, 'Added `day_sin` feature.')

    if 'dow_sin' not in data.columns:
        data['dow_sin'] = np.sin(data.index.dayofweek / data.index.dayofweek.max() * 2 * np.pi)
        log(log_path, 'Added `dow_sin` feature.')

    log(log_path, 'Task 2: Completed adding time-related features.')
    return data

class TargetFeature:
    def __init__(self, log_path, log, results_path, data, test_start, test_end, target='F380 M2/M3',
                 feature='rolling_target5', **kwargs):
        self.log = log
        self.log_path = log_path
        self.results_path = results_path

        self.data = data
        self.test_start = test_start
        self.test_end = test_end
        self.target = target
        self.feature = feature

        # define pd.DataFrame to store resulting df
        self.target_and_feature = None

        # define scaler
        self.scaler = MinMaxScaler()

        # define a starting message for log
        self.message = 'Task 2 [TargetFeature]:\n\t'

    def __call__(self, *args, **kwargs):
        self._setup()

    def _setup(self):
        self.log(self.log_path, f'{self.message}Internal setup for target and feature columns...')

        # get rolling feature
        rolling = self.data[self.target].rolling(5).mean()
        rolling.dropna(inplace=True)
        self.log(self.log_path, f'Created rolling target.')

        # fit a scaler
        self.scaler.fit(rolling[:self.test_end].values.reshape(-1, 1))
        self.log(self.log_path, f'Created a MinMaxScaler for rolling column.')

        # create a dataframe to store current state
        self.target_and_feature = pd.DataFrame({
            'actual_target': self.data[self.target],
            'actual_feature': rolling[:self.test_end],
            'predicted_target': np.nan,
            'predicted_feature': np.nan
        }, index=self.data.index)
        self.log(self.log_path, f'Created a dataframe to store initial actual/predicted target and feature.')

        # fill in predicted target and feature until test_start date
        self.target_and_feature.loc[:self.test_start, 'predicted_target'] = self.data[self.target][:self.test_start]
        self.target_and_feature.loc[:self.test_start, 'predicted_feature'] = rolling[:self.test_start]
        self.log(self.log_path, f'Setting previous predicted target/feature columns to actual values.')
        self.log(self.log_path, f'{self.message}Finish setting up TargetFeature class instance.')

    def get_target_feature(self, end_date, start_date=None, include_past=False):
        # return all values until specified date
        if include_past:
            if start_date is not None:
                self.log(self.log_path, f'Returning dates from {start_date} to {end_date}.')
                return self.target_and_feature.loc[start_date:end_date, :]
            self.log(self.log_path, f'Returning all dates until {end_date}')
            return self.target_and_feature.loc[:end_date, :]

        # return one-day results
        try:
            entry = pd.DataFrame(self.target_and_feature.loc[end_date, :]).T
        except KeyError:
            self.log(self.log_path, f'{self.message}Date {end_date} does not exist!', CRITICAL)
            raise KeyError(f'{self.message}Date {end_date} does not exist!')
        else:
            self.log(self.log_path, f'Returned one-day instance for {end_date}.')
            return entry

    def set_target_feature(self, date, column, value):
        fmt = '%Y-%m-%d'

        # raise error if the supplemented date is less than the test start
        # if datetime.strptime(self.test_start, fmt) < datetime.strptime(date, fmt):
        #     message = f'Illegal date substitution {date}, expected to see dates between {self.test_start} to {date}.'
        #     self.log(self.log_path, self.message + message, CRITICAL)
        #     raise Exception(message)

        # get previous value
        previous = self.target_and_feature.loc[date, column]

        # log information
        self.log(self.log_path, f'{self.message}Setting new value for {column}. Old: {previous}, new: {value}')

        # set new value
        self.target_and_feature.loc[date, column] = value

        # save current state
        # self.target_and_feature.reset_index(inplace=True)
        self.target_and_feature.to_csv(f'{self.results_path}/complete_target_feature_df.csv', index=True)
        self.log(self.log_path, f'{self.message}Saving intermediate results in {self.results_path} CSV file.')



def _setup_experiment(today, target_col, log, date_params):
    path = f'experiments/{today}'
    
    # create dictionary to store experiment info
    setup_dict = dict()
    
    # unpack values
    setup_dict['log_path'] = f'{path}/log.log'
    
    log_path = setup_dict['log_path']
    
    if not os.path.exists(path):
        os.mkdir(path)
        log(log_path, f'Created folder `{path}` to store experiment results.')
    
    setup_dict['save_path'] = path
    setup_dict['model'] = os.path.join(path, 'model')
    setup_dict['pickles'] = os.path.join(path, 'pickles')
    setup_dict['target_col'] = target_col
    setup_dict['train_start'] = date_params['train_start']
    setup_dict['train_end'] = date_params['train_end']
    setup_dict['test_start'] = date_params['test_start']
    setup_dict['test_end'] = date_params['test_end']
    
    return setup_dict

EXP_FOLDER = 'experiments/' + str(datetime.today())[:10]

def _log(fpath, msg, lvl='INFO'):
    now = datetime.now()
    
    with open(fpath, 'a') as fout:
        fout.write(f'[{now}] {lvl.upper()} - {msg}\n')


if __name__ == '__main__':
    # print('Sys path:', sys.path)
    
    date_params = {
        # 'train_start': '2017-03-21',
        # 'train_end': '2023-01-01',
        # 'test_start': '2023-01-01',
        # 'test_end': '2023-03-01'

        'train_start': '2017-03-21',
        'train_end': str(datetime.today())[:10],
        'test_start': str(datetime.today())[:10],
        'test_end': str(datetime.today() + BDay(10))[:10],
    }
    
    ds = str(datetime.today())[:10]
    
    ti = _setup_experiment(ds, 'F380 M2/M3', _log, date_params)
    
    check_data = _check_data(ds, _log, ti)
    
    # ================================== XGBOOST IMPLEMENTATION ==================================
    
    prepare_data = _prepare_data(ds, _log, data_path=check_data, model_type='xgboost', ti=ti)
    
    pretrain_model = _pretrain_model(ds, _log, ti=ti)
    
    get_predictions = _get_predictions(ds, _log, ti=ti)
    
    
    # ================================== VIRTUE IMPLEMENTATION ==================================
    # virtue_data = _prepare_data(ds, _log, data_path=check_data, model_type='virtue', ti=ti)
    # print(virtue_data)
    
    get_predictions = _train_ridge(ds, ti=ti)