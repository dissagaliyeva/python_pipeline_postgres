import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
import graphviz


def iterate(data, idx, step=100, low=False):
    # change step size if we have smaller input data
    step = 50 if low else 100

    # select starting point depending on the size of the input data
    start = idx * step if idx > 0 else 0

    # select ending point (if smaller, 300 days rolling window. Otherwise, 600)
    if low:
        end = 300 + step * idx
    else:
        end = 600 + step * idx

    train, val = data.iloc[start: start + end - 30, :], data.iloc[start + end - 30: start + end, :]
    return train, val


def custom_pnl_10(y_pred, y_true):
    nth = 10

    thresh = 0.2
    try:
        diff = y_pred[nth - 1] - y_pred[0]
    except IndexError:
        diff = y_pred[-1] - y_pred[0]

    direction = 1 if diff > thresh else (-1 if diff < thresh else 0)
    pnl = diff * direction

    return 'pnl', pnl


def show_plot(df, save_path, pred='preds'):
    agg_data = df.copy()
    agg_data.pricing_date = pd.to_datetime(agg_data.pricing_date)
    agg_data.forecast_date = agg_data.forecast_date.astype(str)

    agg_data = agg_data.set_index('pricing_date')

    df = agg_data.copy()
    df = df.sort_index(ascending=True)

    fig = px.line(df, x=df.index, y=pred, color='forecast_date', markers=True, height=600, width=1200)
    fig.update_traces(opacity=0.5, line=dict(width=2, ))

    df = agg_data.copy()
    df = df.sort_index(ascending=True)

    x = df.index
    y = df['target']
    name = 'target'
    fig.add_trace(
        go.Scatter(x=x, y=y, name=name, mode='lines+markers', opacity=0.5, marker=dict(color='darkgray', size=10),
                   line=dict(width=18, color='gray')))  # ,shape = 'hv'
    fig.write_image(save_path)


def show_results(results, save_path):
    thresh = 0.2

    results['nth_day'] = results.groupby('forecast_date')['preds'].cumcount()
    results['preds_diff'] = results.groupby('forecast_date')['preds'].diff()
    results['traget_diff'] = results.groupby('forecast_date')['target'].diff()
    results['preds_diff_cum'] = results.groupby('forecast_date')['preds_diff'].cumsum()
    results['traget_diff_cum'] = results.groupby('forecast_date')['traget_diff'].cumsum()
    results['preds_direct'] = results['preds_diff'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    results['traget_direct'] = results['traget_diff'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    results['preds_direct_cum'] = results['preds_diff_cum'].apply(
        lambda x: 1 if x > thresh else (-1 if x < -thresh else 0))
    results['traget_direct_cum'] = results['traget_diff_cum'].apply(
        lambda x: 1 if x > thresh else (-1 if x < -thresh else 0))
    results['pnl_stra_daily_direct'] = results['preds_direct'] * results['traget_diff']
    results['pnl_stra_cum_direct'] = results['preds_direct_cum'] * results['traget_diff_cum']

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    results.pivot_table(index='pricing_date', values='pnl_stra_cum_direct', aggfunc=sum).cumsum().plot(ax=ax[0],
                                                                                                       title='pnl_stra_cum_direct')
    results.pivot_table(index='pricing_date', values='pnl_stra_daily_direct', aggfunc=sum).cumsum().plot(ax=ax[1],
                                                                                                         title='pnl_stra_daily_direct')
    plt.savefig(save_path)
    plt.close()


def get_train_test(data, columns, train_start, train_end, test_start, test_end):
    # check to verify test_start is after train_end

    # split to train/test
    train = data[(data.index >= train_start) & (data.index < train_end)][columns]
    test = data[(data.index >= test_start) & (data.index < test_end)][columns]

    return train, test


def pretrain_model(model_name, target_feature, results_folder=None, variance=None, ti=None):
    log_path, save_path, target_col = ti['log_path'], ti['save_path'], ti['target_col']
    train_start, train_end, test_start, test_end = ti['train_start'], ti['train_end'], ti['test_start'], ti['test_end']

    # if test_end is None:
    #     try:
    #         test_end = kwargs['test_end']
    #     except KeyError:
    #         log(log_path, 'Task 3: NO TEST_END FOUND (model.pretrain_model @115).', 'CRITICAL')
    #         raise KeyError('Task 3: NO TEST_END FOUND (model.pretrain_model @115).')

    # get train, test
    results = pickle.load(open(f'{results_folder}/pickles/results_variance={variance}', 'rb'))
    train, test = results['train'], results['test']

    # check if model exists
    if os.path.exists(model_name):
        # log(log_path, f'Task 3: Loading existing model {model_name}...')

        # load existing model
        model = xgb.Booster()
        model.load_model(model_name)

        return model

        # otherwise, pretrain the model from scratch
    fmt = '%Y-%m-%d'

    # set smaller step size if less data is selected
    low = datetime.strptime(train_end, fmt) < datetime.strptime('2021-01-01', fmt)

    # get names for target_feature columns
    rolling, target = 'actual_feature', 'actual_target'

    # set parameters
    params = {'process_type': 'default', 'refresh_leaf': True, 'min_child_weight': 7,
              'subsample': 1, 'colsample_bytree': 0.5, 'eta': 0.03}

    # start pretraining the model
    for idx in range(300):
        # get batches
        train_temp, valid_temp = iterate(train, idx, low=low)

        # break if there are fewer values
        if len(train_temp) < 200 or len(valid_temp) <= 5: break

        # get rolling and target columns
        train_temp[[target_col, 'rolling_target5']] = \
            target_feature.get_target_feature(end_date=train_temp.index.max(), start_date=train_temp.index.min(),
                                              include_past=True)[[target, rolling]].values
        valid_temp[[target_col, 'rolling_target5']] = \
            target_feature.get_target_feature(end_date=valid_temp.index.max(), start_date=valid_temp.index.min(),
                                              include_past=True)[[target, rolling]].values

        train_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_temp.dropna(inplace=True)
        valid_temp.dropna(inplace=True)

        # normalize rolling
        train_temp['rolling_target5'] = target_feature.scaler.transform(train_temp[['rolling_target5']])
        valid_temp['rolling_target5'] = target_feature.scaler.transform(valid_temp[['rolling_target5']])

        train_dm = xgb.DMatrix(train_temp.drop(columns=target_col), label=train_temp[target_col])
        valid_dm = xgb.DMatrix(valid_temp.drop(columns=target_col), label=valid_temp[target_col])

        if idx == 0:
            model = xgb.train(params, train_dm, 100, evals=[(valid_dm, 'valid')],
                              maximize=True, early_stopping_rounds=10, custom_metric=custom_pnl_10)
        else:
            model = xgb.train(params, train_dm, 100, evals=[(valid_dm, 'valid')],
                              maximize=True, early_stopping_rounds=10,
                              xgb_model=model_name, custom_metric=custom_pnl_10)

        model.save_model(model_name)

    # final refit
    full_train = train.copy()
    full_train[[target_col, 'rolling_target5']] = target_feature.get_target_feature(end_date=train.index.max(),
                                                                                    include_past=True)[[target, rolling]].values
    full_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_train.dropna(inplace=True)

    # normalize data
    full_train['rolling_target5'] = target_feature.scaler.transform(full_train[['rolling_target5']])

    train_dm = xgb.DMatrix(full_train.drop(columns=target_col), label=full_train[target_col])

    model = xgb.train(params, train_dm, evals=[(train_dm, 'train')],
                      maximize=True, early_stopping_rounds=10,
                      xgb_model=model_name, custom_metric=custom_pnl_10)

    model.save_model(model_name)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    xgb.plot_tree(model, ax=ax)

    # Save the plot as an image file (e.g., PNG or PDF)
    output_file_path = f"{results_folder}/tree_plot.png"
    fig.savefig(output_file_path)

    return model


def get_predictions(data, columns, target_feature, model, results_folder, variance,
                    train_start, train_end, test_start, test_end, log, log_path):
    results = pickle.load(open(f'{results_folder}/pickles/results_variance={variance}', 'rb'))
    train, test = results['train'], results['test']

    test.dropna(inplace=True)

    log(log_path, 'Task 4: `get_predictions` function activated...')
    log(log_path, f'Task 4: Train head: {train.head()}')
    log(log_path, f'Task 4: Number of columns: {len(columns)}. Columns: {columns}')

    scaler = results['feature_scaler']

    window_size = 10
    results_df = pd.DataFrame(columns=['pricing_date', 'preds', 'target', 'forecast_date'])
    market_data = data.copy()[columns]

    # iterate over values
    for counter in range(len(test)):
        try:
            sliding_history = market_data[market_data.index >= test_start].iloc[counter: counter + window_size, :]
        except IndexError:
            pass
        else:
            # verify the length of the sliding history == window_size
            if len(sliding_history) < window_size:
                break

            # copy original data
            full_history = market_data.copy()

            # keep track of every 0th value to select all actual targets and then predictions afterward
            first_day, second_day = None, None

            for counter2 in range(window_size):
                try:
                    sliding_start_date = pd.DataFrame(sliding_history.iloc[counter2, :]).T.index.values[0]

                    if counter2 == 1:
                        second_day = sliding_start_date

                except IndexError:
                    pass
                else:
                    print('Current Sliding start date:', sliding_start_date)
                    expanding_history = full_history[full_history.index <= sliding_start_date]

                    # get test value and convert to DMatrix
                    testx = pd.DataFrame(expanding_history.iloc[-1, :]).T

                    # normalize features
                    testx = pd.DataFrame(scaler.transform(testx), columns=testx.columns, index=testx.index)

                    # select current state of the target/feature dataframe
                    current_values = target_feature.get_target_feature(sliding_start_date, include_past=True)
                    print('@244 Current values tail:', current_values[['actual_target', 'actual_feature']].tail(3))
                    print('@245 Current values tail:', current_values[['predicted_target', 'predicted_feature']].tail(3))

                    # select actual values for the first predictions
                    if counter2 == 0:
                        current_targets = current_values[['actual_target']]
                    else:
                        current_targets = current_values['actual_target'].copy()
                        current_targets.loc[second_day:sliding_start_date] = current_values['predicted_target'].loc[
                                                                             second_day:sliding_start_date]

                    # get rolling targets
                    rolling_target = current_targets.rolling(5).mean()
                    rolling_target.dropna(inplace=True)
                    print('Rolling target', rolling_target.tail())

                    # normalize rolling targets
                    try:
                        rolling_target = pd.DataFrame(target_feature.scaler.transform(rolling_target),
                                                      index=rolling_target.index)
                    except ValueError:
                        rolling_target = pd.DataFrame(
                            target_feature.scaler.transform(rolling_target.values.reshape(-1, 1)),
                            index=rolling_target.index)

                    # put the new value to test set
                    try:
                        print('Rolling last values', rolling_target.tail(5))
                        testx['rolling_target5'] = rolling_target.iloc[-1, 0]
                    except KeyError:
                        pass
                    else:
                        print('Current test day:', testx, end='\n=====\n\n')
                        # convert test to DMatrix
                        test_dm = xgb.DMatrix(testx)

                        # get predictions
                        prediction = model.predict(test_dm)

                        # append new prediction to predicted target column
                        target_feature.set_target_feature(str(sliding_start_date)[:10], 'predicted_target', prediction)

                        # append new rolling value to the predicted feature column
                        inverse = target_feature.scaler.inverse_transform(testx[['rolling_target5']])
                        print('Inverse @288:', inverse)
                        target_feature.set_target_feature(str(sliding_start_date)[:10], 'predicted_feature', inverse)
                        
                        print(results_df, end='\n\n\n')

                        # record results
                        results_df = pd.concat([results_df, pd.DataFrame({
                            'pricing_date': testx.index,
                            'preds': prediction,
                            'target': current_values.loc[sliding_start_date, 'actual_target'],
                            'forecast_date': sliding_history.index.min()

                        })], ignore_index=True)

    results_df.to_csv(f'{results_folder}/results-for-{test_start}-{test_end}.csv')
    show_results(results_df, f'{results_folder}/daily_pnl-for-{test_start}-{test_end}.png')
    show_plot(results_df, f'{results_folder}/plot.png')

 