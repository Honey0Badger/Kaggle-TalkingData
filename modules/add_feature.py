import sys
import gc
import pandas as pd
import random
import numpy as np
import math

# helper functions and feature engineering functions
def load_from_file(filepath):
        data = pd.read_csv(filepath, parse_dates=['click_time'])
        print ("Loading full data finished...")
        return data

def load_half_file(filepath):
        data = pd.read_csv(filepath, skiprows=range(1,124903891)
                , nrows=60000000, parse_dates=['click_time'])
        print ("Loading part data finished...")
        return data

# feature lists:

def do_os_click_freq( train_df, feature_name ):
    gp = train_df[['ip', 'os']].groupby(by=['os'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': feature_name})
    return gp

def do_app_os_click_freq( train_df, feature_name  ):
    gp = train_df[['ip', 'app', 'os']].groupby(by=['app', 'os'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': feature_name})
    return gp

def do_chn_os_click_freq( train_df, feature_name  ):
    gp = train_df[['ip', 'channel', 'os']].groupby(by=['channel', 'os'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': feature_name})
    return gp

def do_chn_dev_click_freq( train_df, feature_name  ):
    gp = train_df[['ip', 'channel', 'device']].groupby(by=['channel', 'device'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': feature_name})
    return gp

def do_app_dev_click_freq( train_df, feature_name  ):
    gp = train_df[['ip', 'app', 'device']].groupby(by=['app', 'device'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': feature_name})
    return gp

def ip_app_next_click( full_df, feature_name ):
    gp = full_df[['ip', 'app', 'click_time']]\
            .groupby(by=['ip', 'app'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    return gp

def ip_os_next_Click( full_df, feature_name ):
    gp = full_df[['ip', 'os', 'click_time']]\
            .groupby(by=['ip', 'os'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    return gp

def ip_chn_nextClick( full_df, feature_name ):
    gp = full_df[['ip', 'channel', 'click_time']]\
            .groupby(by=['ip', 'channel'])\
            .click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
# main part

feature_name = 'ip_os_nextClick'


train_file = '../input/train.csv'
test_file = '../input/test.csv'
train_df = load_half_file(train_file)
len_train = len(train_df)
test_df = load_from_file(test_file)
full_df = train_df.append(test_df)

del train_df, test_df
gc.collect()

feature_df = ip_app_next_click( full_df, feature_name );

feature_df.to_csv("./features/"+feature_name+".csv"\
                  , index=False, float_format='%1.5f')

