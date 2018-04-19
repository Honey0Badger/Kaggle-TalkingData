import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd

from preprocess2 import *
from models import *
from pipelines import *

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

debug = False
################### load raw data ##################################
#
#if debug:
#    train_file = '../input/train_debug.csv'
#    test_file = '../input/test_debug.csv'
#else:
#    train_file = '../input/train.csv'
#    test_file = '../input/test.csv'
#
#full_df, len_train, predictors = read_merge_process2(train_file, ftest=test_file)
#print("*************************  Full data info **********************************\n")
#full_df.info()
#print("*************************  End of data info **********************************\n")
#sys.stdout.flush()
#            
#process = psutil.Process(os.getpid())
#print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
#sys.stdout.flush()
#
#print('all columns:\n', list(full_df), "\n")
#target = 'is_attributed'
#
#print('Predictors used for training: \n', predictors, "\n")
#cat_features = ['app', 'device','os', 'channel', 'day', 'hour']
#sys.stdout.flush()
################## end of loading and processing raw data #######

##################### load pre-processed data #####################
full_df = pd.read_pickle('22_feature_bot60m_data.pkl')

predictors = [ 'app','channel','device','os','day','hour',
               'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 
               'X7', 'X8', 'ip_tcount', 'ip_app_count', 
               'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
               'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
               'nextClick', 'nextClick_shift' ]
cat_features = ['app', 'device','os', 'channel', 'day', 'hour']

len_train = 60000000
target = 'is_attributed'

print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
#################### end of loading processed data ###################################

print('total train data size: ', len_train)
#train_split = 170000000
train_split = 56000000
#train_split = 40000
train_df = full_df[:train_split]
valid_df = full_df[train_split:len_train]
test_df = full_df[len_train:]
del full_df
gc.collect()
print('train size: ', len(train_df))
print('valid size: ', len(valid_df))
print(' test size: ', len(test_df))
print('\n')
sys.stdout.flush()
train_dataset, valid_dataset = LGBM_DataSet(train_df, valid_df, predictors, target, cat_features)

process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

kernal_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_iterations': 1000,
        'learning_rate': 0.20,
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200, # because training data is extremely unbalanced 
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        }

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_iterations': 1000,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 25,  # 2^max_depth - 1
    'max_depth': 12,  # -1 means no limit
    'min_child_samples': 70,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 400,  # Number of bucketed bin for feature values
    'subsample': 0.65,  # Subsample ratio of the training instance.
    #'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 1.0,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 1e-3,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight':300 # because training data is extremely unbalanced 
}

print("start training...\n")
print("model param: ", lgbm_params)
print("\n")
sys.stdout.flush()
lgbm_model = single_LGBM_train(lgbm_params, 
                               train_dataset, 
                               valid_dataset, 
                               metrics='auc',
                               early_stopping_rounds=30)

process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
del train_dataset, valid_dataset
gc.collect()
sys.stdout.flush()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = lgbm_model.predict(test_df[predictors])
if debug:
    sub.to_csv("./debug_sub.csv", index=False, float_format='%1.5f')
else:
    sub.to_csv("../output/lgbm_f22_SM_bot50m_gridCV_param.csv", index=False, float_format='%1.5f')

