import pickle
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, inspect

import helpers


def connect_to_db():
    host = 'host.docker.internal'
    database = 'postgres'
    user = 'postgres'
    password = 's58me19dax'
    connection_string = f"postgresql://{user}:{password}@{host}/{database}"

    engine = create_engine(connection_string)
    return engine


def check_db(engine, table_name='raw'):
    insp = inspect(engine)
    exists = table_name in insp.get_table_names()
    
    if exists:
        # get last date from the table 
        sql_df = pd.read_sql(f'SELECT * FROM {table_name}', engine)
        
        # convert to datetime
        if 'order_date' in sql_df.columns:
            sql_df.order_date = pd.to_datetime(sql_df.order_date)
            last_date = sql_df.order_date.max()
        else:
            sql_df.pricing_date = pd.to_datetime(sql_df.pricing_date)
            last_date = sql_df.pricing_date.max()
        
        return last_date
    else:
        print('Current table:', table_name)
        df_path = 'Pickle/'

        if table_name == 'last_order':
            df_path += 'last_order-2017-01-01-2023-10-09.pickle'
        elif table_name == 'asia' or table_name == 'raw_data':
            df_path += 'Asia_EU_FO_PVO_2016-2023.pickle'
        elif table_name.startswith('pvo') or table_name.startswith('lo_expanded'):
            df_path += 'pvo_expanded-2017-01-01-2023-10-09.pickle'
        else:
            raise KeyError('Accessing wrong dataset, @db_operations.py, line 45')
        
        df = pickle.load(open(df_path, 'rb'))

        # create new table 
        df.to_sql(table_name, engine, if_exists='replace', index=False)

        if 'order_date' in df.columns:
            df.order_date = pd.to_datetime(df.order_date)
            return  df.order_date.max()
        

def add_to_db(engine, df, today_date, table_name='raw'):
    # Step 1: load the dataset from the db
    # engine = connect_to_db() if engine is None else engine
    
    date = check_db(engine, table_name)
    
    print('LAST_DATE:', today_date)
    print('DATE:', date)

    # Otherwise, append to the table 
    if datetime.strptime(today_date, '%Y-%m-%d') > datetime.strptime(str(date)[:10], '%Y-%m-%d'):
        if 'order_date' in df.columns:
            df = df[(df.order_date > date) & (df.order_date <= today_date)]
        elif 'pricing_date' in df.columns:
            df = df[(df.pricing_date > date) & (df.pricing_date <= today_date)]
            
        df.to_csv(f'save={table_name}.csv')
        
        orig_data = pd.read_sql(f'SELECT * FROM {table_name}', engine)
        
        orig_data = pd.concat([orig_data, df], ignore_index=False)
        orig_data.drop_duplicates(inplace=True)
        
        if 'pricing_date' in orig_data.columns:
            orig_data.drop(index=orig_data[orig_data.pricing_date.isna()].index, inplace=True)
        elif 'order_date' in orig_data.columns:
            orig_data.drop(index=orig_data[orig_data.order_date.isna()].index, inplace=True)
        
        orig_data.to_sql(table_name, engine, if_exists='replace', index=False)
        
        
        # print('Appending dates:', df.index)
        # df.to_sql(table_name, engine, if_exists='append', index=False)