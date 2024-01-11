import matplotlib.pyplot as plt
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
import pickle
from sklearn.preprocessing import MinMaxScaler

import billiard as multiprocessing


def preprocess_data(today, log, **kwargs):
    ti = kwargs['ti'].xcom_pull(task_ids='setup_dir')
    log_path, save_path, target = ti['log_path'], ti['save_path'], ti['target_col']
    train_start, train_end, test_start, test_end = ti['train_start'], ti['train_end'], ti['test_start'], ti['test_end']
    
    data_file = 'data/data_' + today + '.csv'

    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
    else:
        try:
            data = pickle.load(open(data_file, 'rb'))
        except TypeError:
            log(log_path, 'No data was pushed!', 'CRITICAL')
            raise TypeError('No data was created!')
    
    features, targets = data.drop(columns=target), data[target]
    features.reset_index(inplace=True)
    extracted = extract_features(features, column_sort='pricing_date', column_id='pricing_date')
    impute(extracted)
    features_filtered = select_features(extracted, targets)
    
    X_train, X_test = features_filtered[features_filtered.index < test_start], features_filtered[(features_filtered.index >= test_start) & (features_filtered.index <= test_end)]
    y_train, y_test = targets[targets.index < test_start], targets[(targets.index >= test_start) & (targets.index <= test_end)]
    
    pickle_loc = f'{save_path}/pickles'
    pickle_file = f'{pickle_loc}/tsfresh_data'

    # create a folder to store results
    if not os.path.exists(pickle_loc): os.mkdir(pickle_loc)

    # store results
    with open(pickle_file, 'wb') as fout:
        pickle.dump({'features_filtered': features_filtered, 'train': X_train, 'test': X_test,
                     'y_train': y_train, 'y_test': y_test, 'y': targets},
                    fout, pickle.HIGHEST_PROTOCOL)

    return pickle_file


def train_ridge(z=[10.000], **kwargs):
    ti = kwargs['ti'].xcom_pull(task_ids='setup_dir')
    save_path = ti['save_path'] 
    
    ti = kwargs['ti'].xcom_pull(task_ids='preprocess_data')
    
    try:
        file = pickle.load(open(ti, 'rb'))
    except FileExistsError:
        pass
    else:
        features_filtered, train, test, y = file['features_filtered'], file['train'], file['test'], file['y']
        
        results_df = pd.DataFrame(columns=['pricing_date', 'preds', 'target'])
        
        scaler = MinMaxScaler().fit(train)
        y_scaler = MinMaxScaler().fit(y[y.index < test.index.min()])
        
        for z_val in z:
            for date in test.index.tolist():
                date = str(date)[:10]
                
                full_history = features_filtered.copy()
                X_train = scaler.transform(full_history[full_history.index < date])
                y_train = y_scaler.transform(y[y.index < date])[:, 0]
                print('Train:', X_train.shape, y_train.shape)
                
                X_test, y_test = scaler.transform(full_history[full_history.index >= date]), y_scaler.transform(y[y.index >= date])[:, 0]
                print('Test:', X_test.shape, y_test.shape)
                
                beta = Ridge(alpha=z_val, solver='svd', fit_intercept=False).fit(X_train, y_train).coef_
                
                forecast = X_test @ beta
                
                results_df = pd.concat([results_df, pd.DataFrame({
                    'pricing_date': [date],
                    'preds': y_scaler.inverse_transform(np.array([forecast]).reshape(-1, 1))[0],
                    'target': y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1))[0],
                })], ignore_index=True)
                            
                            
            results_df.to_csv(os.path.join(save_path, f'tsfresh_virtue_z={z_val}.csv'))