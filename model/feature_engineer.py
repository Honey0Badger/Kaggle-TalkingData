"""
The following functions are an ensemble of engineered features from Kaggle talkingData public
discussion forum. Not all of them are used for the final model.
"""

import pandas as pd
import random
import numpy as np

import gc

def do_count( df, group_cols, col_name):
    print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size()\
         .rename(col_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[col_name] = df[col_name].astype('uint32')
    feat_col = df[[col_name]] 
    df.drop([col_name], axis=1, inplace=True)
    gc.collect()
    return feat_col

def do_countuniq( df, group_cols, counted, col_name):
    print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted]\
            .nunique().reset_index().rename(columns={counted:col_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    print( col_name + " max value = ", df[col_name].max() )
    df[col_name] = df[col_name].astype('uint32')
    feat_col = df[[col_name]] 
    df.drop([col_name], axis=1, inplace=True)
    return feat_col

def do_cumcount( df, group_cols, counted, col_name):
    print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[col_name]=gp.values
    del gp
    df[col_name] = df[col_name].astype('uint32')
    feat_col = df[[col_name]] 
    df.drop([col_name], axis=1, inplace=True)
    gc.collect()
    return feat_col


def do_mean( df, group_cols, counted, col_name):
    print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean()\
            .reset_index().rename(columns={counted:col_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[col_name] = df[col_name].astype('float32')
    feat_col = df[[col_name]] 
    df.drop([col_name], axis=1, inplace=True)
    return feat_col

def do_var( df, group_cols, counted, col_name):
    print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var()\
            .reset_index().rename(columns={counted:col_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[col_name] = df[col_name].astype('float32')
    feat_col = df[[col_name]] 
    df.drop([col_name], axis=1, inplace=True)
    return feat_col

def do_app_click_freq( train_df ):
    col_name =  'app_click_freq'
    gp = train_df[['ip', 'app']].groupby(by=['app'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': col_name})
    train_df = train_df.merge(gp, on=['app'], how='left')
    del gp
    gc.collect()
    feat_col = train_df[[col_name]] 
    train_df.drop([col_name], axis=1, inplace=True)
    return feat_col

def do_nextClick( train_df ):
    print('doing nextClick...')
    col_name = 'nextClick'
    D=2**26
    train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app']\
            .astype(str) + "_" + train_df['device'].astype(str) \
                            + "_" + train_df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)

    next_clicks= []
    for category, t in zip(reversed(train_df['category'].values)
                           , reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
    del(click_buffer)
    QQ= list(reversed(next_clicks))
    train_df[col_name] = pd.Series(QQ).astype('float32')

    feat_col = train_df[[col_name]] 
    train_df.drop([col_name], axis=1, inplace=True)
    return feat_col
