from sklearn import preprocessing

import sys
import gc
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def read_merge_process2(ftrain, ftest=None):
    if (ftest != None):
        train_df = load_half_file(ftrain)
        #train_df = load_from_file(ftrain)
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

    # feature engineering
    train_df['day'] = train_df.click_time.dt.day.astype('uint8')
    train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')

    gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
    train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
    train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True ); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True ); gc.collect()
    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()

    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'

    D=2**26
    train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
                            + "_" + train_df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)

    train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks= []
    for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
    del(click_buffer)
    QQ= list(reversed(next_clicks))

    train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)

    train_df[new_feature] = pd.Series(QQ).astype('float32')
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
    predictors.append(new_feature+'_shift')
    
    del QQ
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
    
    return train_df, len_train, predictors


def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def load_from_file(filepath):
        data = pd.read_csv(filepath, parse_dates=['click_time'])
        print ("Loading full data finished...")
        return data

def load_half_file(filepath):
        data = pd.read_csv(filepath, skiprows=range(1,124903891)
                , nrows=60000000, parse_dates=['click_time'])
        print ("Loading part data finished...")
        return data

def load_file_cv(filepath):
        data = pd.read_csv(filepath, skiprows=range(1,174903891)
                , nrows=10000000, parse_dates=['click_time'])
        #data = pd.read_csv(filepath)
        print ("Loading part data finished...")
        return data
