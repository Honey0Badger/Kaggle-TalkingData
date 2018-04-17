import xgboost as xgb

from scipy import sparse
from sklearn.model_selection import train_test_split
from scipy.stats import skew, boxcox
from sklearn import preprocessing

import sys
import gc
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def read_merge_process(ftrain, ftest=None):
    if (ftest != None):
        train_df = load_from_file(ftrain)
        len_train = len(train_df)
        test_df = load_from_file(ftest)
        train_df = train_df.append(test_df)
        del test_df
    else:
        train_df = load_file_cv(ftrain)
        len_train = len(train_df)
    gc.collect()

    print('Preparing data...')
    sys.stdout.flush()

    # added features on 04/13/18
    train_df['ip_nextClick'] = train_df[['ip','click_time']].groupby(by=['ip'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    train_df['ip_app_nextClick'] = train_df[['ip', 'app', 'click_time']]\
            .groupby(by=['ip', 'app'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    train_df['ip_chn_nextClick'] = train_df[['ip', 'channel', 'click_time']]\
            .groupby(by=['ip', 'channel'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    train_df['ip_os_nextClick'] = train_df[['ip', 'os', 'click_time']]\
            .groupby(by=['ip', 'os'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    gc.collect()

    train_df['weekday'] = train_df.click_time.dt.day.astype('uint8')
    train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')
    train_df.drop(['click_time', 'attributed_time'], axis=1, inplace=True)
    gc.collect()

    print('adding features...')
    sys.stdout.flush()

    # features added on 04/09/18
    gp = train_df[['ip','app', 'channel']]\
            .groupby(by=['ip', 'app'])[['channel']].count().reset_index()\
            .rename(index=str, columns={'channel': 'ip_app_count_chns'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    gp = train_df[['ip', 'app']].groupby(by=['app'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': 'app_click_freq'})
    train_df = train_df.merge(gp, on=['app'], how='left')
    del gp
    gc.collect()

    gp = train_df[['app', 'channel']].groupby(by=['app'])[['channel']]\
                   .count().reset_index().rename(index=str, columns\
                   ={'channel': 'app_freq'})
    train_df = train_df.merge(gp, on=['app'], how='left')

    gp = train_df[['app', 'channel']].groupby(by=['channel'])[['app']]\
                   .count().reset_index().rename(index=str, columns\
                   ={'app': 'channel_freq'})
    train_df = train_df.merge(gp, on=['channel'], how='left')
    del gp
    gc.collect()

    # added features on 04/11/18
    gp = train_df[['ip','weekday','hour','channel']]\
            .groupby(by=['ip','weekday','hour'])[['channel']].count()\
            .reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count_chns'})
    train_df = train_df.merge(gp, on=['ip','weekday','hour'], how='left')
    del gp
    gc.collect()

    gp = train_df[['ip','app', 'os', 'channel']]\
            .groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index()\
            .rename(index=str, columns={'channel': 'ip_app_os_count_chns'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    gp = train_df[['ip','weekday','hour','channel']].groupby(by=['ip','weekday','channel'])\
            [['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_day_chn_var_hour'})
    train_df = train_df.merge(gp, on=['ip','weekday','channel'], how='left')
    train_df.fillna(0, inplace=True)
    del gp
    gc.collect()

    gp = train_df[['ip','app', 'channel','hour']]\
            .groupby(by=['ip', 'app', 'channel'])[['hour']].mean()\
            .reset_index().rename(index=str, columns={'hour': 'ip_app_chn_mean_hour'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()


    train_df['ip_app_count_chns'] = train_df['ip_app_count_chns'].astype('uint16')
    train_df['ip_day_hour_count_chns'] = train_df['ip_day_hour_count_chns'].astype('uint16')
    train_df['ip_app_os_count_chns'] = train_df['ip_app_os_count_chns'].astype('uint16')

    predictors = ['app','device','os','channel','weekday','hour', 
                  'ip_app_count_chns', 'app_click_freq','app_freq',
                  'channel_freq',  'ip_day_hour_count_chns', 
                  'ip_app_os_count_chns',  'ip_day_chn_var_hour',
                  'ip_app_chn_mean_hour', 'ip_nextClick',  'ip_app_nextClick',
                  'ip_chn_nextClick', 'ip_os_nextClick' ]
    
    return train_df, len_train, predictors



def load_from_file(filepath):
        data = pd.read_csv(filepath, parse_dates=['click_time'])
        print ("Loading full data finished...")
        return data

def load_half_file(filepath):
        data = pd.read_csv(filepath, skiprows=range(1,134903891)
                , nrows=50000000, parse_dates=['click_time'])
        print ("Loading part data finished...")
        return data

def load_file_cv(filepath):
        #data = pd.read_csv(filepath, skiprows=range(1,174903891)
        #        , nrows=10000000, parse_dates=['click_time'])
        data = pd.read_csv(filepath, parse_dates=['click_time'])
        print ("Loading part data finished...")
        return data

def sample_inputs(infile, outfile):
        data = pd.read_csv(infile)
        sample = data.head(5000)
        sample.to_csv(outfile, index=False)

def train_DataFrame_processed(train):
        train_ts = pd.to_datetime(train['click_time'])
        train.drop(['click_time', 'attributed_time'], axis=1, inplace=True)
        gc.collect()
        train = train.assign(weekday=train_ts.dt.day.astype('uint8'))
        train = train.assign(hour=train_ts.dt.hour.astype('uint8'))
        del train_ts
        gc.collect()
        return train

def test_DataFrame_processed(test):
        test_ts = pd.to_datetime(test['click_time'])
        test.drop(['click_id', 'click_time']
                    , axis=1, inplace=True)
        test = test.assign(weekday=test_ts.dt.day.astype('uint8'))
        test = test.assign(hour=test_ts.dt.hour.astype('uint8'))
        del test_ts
        gc.collect()

        return test

def process_trainData(train):
        train_ts = pd.to_datetime(train['click_time'])
        train_y = train.is_attributed
        train.drop(['click_time', 'attributed_time', 'is_attributed']
                    , axis=1, inplace=True)
        gc.collect()
        train = train.assign(weekday=train_ts.dt.day.astype('uint8'))
        train = train.assign(hour=train_ts.dt.hour.astype('uint8'))
        del train_ts
        gc.collect()
        return train.values, train_y.values
        
def process_testData(test):
        test_ts = pd.to_datetime(test['click_time'])
        test.drop(['click_id', 'click_time']
                    , axis=1, inplace=True)
        gc.collect()
        test = test.assign(weekday=test_ts.dt.day.astype('uint8'))
        test = test.assign(hour=test_ts.dt.hour.astype('uint8'))
        del test_ts
        gc.collect()

        return test.values

def data_features(data):
        data_types = data.dtypes
        cat_cols = list(data_types[data_types=='object'].index)
        num_cols = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)
        id_col = 'id'
        target_col = 'loss'
        num_cols.remove('id')
        num_cols.remove('loss')

        print ( "Categorical features:", cat_cols)
        print ( "Numerica features:", num_cols)
        print ( "ID: %s, target: %s" %( id_col, target_col))
        return id_col, target_col, cat_cols, num_cols
        

def category_encoding(data, cat_cols):
        for cat_col in cat_cols:
                print ("Factorize feature %s" % (cat_col))
                data[cat_col] = preprocessing.LabelEncoder().fit_transform(data[cat_col])
        print ('Label enconding finished...')
        return data
        

def one_hot_encoding(data, cat_cols):
        OHE = preprocessing.OneHotEncoder(sparse=True)
        data_sparse = OHE.fit_transform(data[cat_cols])
        print ('One-hot-encoding finished...')
        print ('data sparse shape:', data_sparse.shape)
        return data_sparse

def process_num_data(data, num_cols):
        skewed_cols = data[num_cols].apply(lambda x: skew(x.dropna()))
        print(skewed_cols.sort_values())

        #** Apply box-cox transformations: **
        skewed_cols = skewed_cols[skewed_cols > 0.25].index.values
        for col in skewed_cols:
                data[col], lam = boxcox(data[col] + 1)

        #** Apply Standard Scaling:**
        for col in num_cols:
                data[[col]] = preprocessing.StandardScaler().fit_transform(data[[col]])
        return data

def train_valid_split(train_x, train_y):
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=0.8, random_state=1234)
        xgtrain = xgb.DMatrix(train_x, label=train_y, missing=np.nan)
        return X_train, X_val, y_train, y_val, xgtrain

        



