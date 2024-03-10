import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import param
import panel
warnings.filterwarnings('ignore')


def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def get_model(path1, path2):
    return load_model(path1), load_model(path2)


def main(result1, model1, result2, model2, orig_data, use_rolling, use_feature_eng):
    assert isinstance(use_rolling, list), '`use_rolling` argument should be a list!'
    
    data1 = get_preds_df(result1)
    _, columns1 = prepare_data(orig_data, rolling=use_rolling[0], feature_eng=use_feature_eng[0])
    
    data2 = get_preds_df(result2)
    _, columns2 = prepare_data(orig_data, rolling=use_rolling[1], feature_eng=use_feature_eng[1])
    
    orig_data, _ = prepare_data(orig_data, rolling=True, feature_eng=False)
    
    model1, model2 = get_model(model1, model2)
    
    print(model1, model2)
    
    shap1, shap2 = get_shap_values(orig_data[columns1], model1, type='other'), get_shap_values(orig_data[columns2], model2, type='other')
    
    return {'results1': [data1, columns1, model1, shap1],
            'results2': [data2, columns2, model2, shap2],
            'orig_data': orig_data}
    
    
def get_shap_values(df, model, type='default'):
    if type == 'default':
        explainer = shap.TreeExplainer(model)
        return explainer, explainer.shap_values(df)
    
    explainer = shap.Explainer(model)
    return explainer(df)


def prepare_data(data, rolling=False, train_start='2017-03-21', test_start='2023-01-01', 
                 test_end='2023-03-01', target='F380 M2/M3', feature_eng=False):
    
    if isinstance(data, str):
        data = pd.read_csv(data)
    
    # load data
    data.pricing_date = pd.to_datetime(data.pricing_date)
    data.set_index('pricing_date', inplace=True)

    # slice the dataset
    # data = data[data.index < non_consecutive_days(data[[target]])]
    data.interpolate(method='linear', inplace=True)
    
    data = data[(data.index >= train_start) & (data.index <= test_end)]
    data[target] = data[target].fillna(0)
    data.dropna(inplace=True)
    
    target_df = data[[target]]
    data.drop(columns=target, inplace=True)
    
    data = add_time_features(data)
    
    if feature_eng:
        columns = drop_low_variance(data, test_start, test_end).tolist()
    else:
        columns = data.columns.tolist()
    
    if rolling:
        # add rolling target
        columns += ['rolling_target5']
        data['rolling_target5'] = target_df.rolling(5).mean()
    
    data[target] = target_df.values
    data.dropna(inplace=True)
    
    return data.loc[:test_end], columns
    


def add_time_features(data):
    if 'year_sin' not in data.columns:
        data['year_sin'] = np.sin(data.index.year / data.index.year.max() * 2 * np.pi)

    if 'month_sin' not in data.columns:
        data['month_sin'] = np.sin(data.index.month / data.index.month.max() * 2 * np.pi)

    if 'day_sin' not in data.columns:
        data['day_sin'] = np.sin(data.index.day / data.index.day.max() * 2 * np.pi)

    if 'dow_sin' not in data.columns:
        data['dow_sin'] = np.sin(data.index.dayofweek / data.index.dayofweek.max() * 2 * np.pi)

    return data


def drop_low_variance(data, test_start, test_end, var=0.2):
    # remove rows after test start
    data = data[(data.index > '2017-03-21') & (data.index < test_start)]

    # check the length of prediction dates
    if len(data[(data.index >= test_start) & (data.index < test_end)]) < 100:
        return data.columns[data.var() > var]

    return data[['_rgp_01_sz_3', '_rgp_02_sz_3', '_rgp_03_sz_3', '_rgp_04_sz_3', '_rgp_05_sz_3',
                 'BPSG', 'COASTAL', 'GUNVORSG', 'HL', 'MERCURIASG', 'P66SG', 'PETROCHINA',
                 'SIETCO', 'TOTALSG', 'TRAFI', 'VITOLSG']].columns
    


def custom_pnl_10(y_pred, y_true):
    nth = len(y_pred)

    thresh = 0.2
    
    try:
        diff = y_pred[-1] - y_pred[0]
    except IndexError:
        diff = y_pred[-1] - y_pred[0]

    direction = 1 if diff > thresh else (-1 if diff < thresh else 0)
    pnl = diff * direction

    return 'pnl', pnl


def get_preds(preds, date):
    preds_df = preds.copy()
    preds_df = preds_df[preds_df['forecast_date'] == date]

    if 'index' in preds_df.columns:
        idx = np.unique(preds_df['index'], return_index=True)[1]
    else:
        idx = np.unique(preds_df.index, return_index=True)[1]
    return preds_df.iloc[idx]


def get_preds_df(path):
    preds = pd.read_csv(path, index_col=0)
    preds.set_index('pricing_date', inplace=True)
    preds.index = pd.to_datetime(preds.index)
    preds.drop(columns='target', inplace=True)
    return preds


def get_dates_between(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Calculate the difference between end_date and start_date
    delta = end_date - start_date
    
    # Generate a list of dates within the specified range
    date_list = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    
    # Format the dates as strings
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_list]
    
    return formatted_dates

