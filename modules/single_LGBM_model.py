import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd

#from preprocess3 import *
from feature_test import *
from models import *
from pipelines import *

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

debug = False
#################### load raw data ##################################
##
#if debug:
#    train_file = '../input/train_debug.csv'
#    test_file = '../input/test_debug.csv'
#else:
#    train_file = '../input/train.csv'
#    test_file = '../input/test.csv'
#
#full_df, len_train, predictors = read_merge_process3(train_file, ftest=test_file)
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
#cat_features = ['app', 'device','os', 'channel', 'hour']
#sys.stdout.flush()
################## end of loading and processing raw data #######

##################### load pre-processed data #####################
full_df = pd.read_pickle('26_feature_bot60m_data.pkl')

predictors = ['nextClick', 'app','device','os', 'channel', 'hour', 
                  'app_click_freq', 'app_os_click_freq', 'app_dev_click_freq',
                  'chn_os_click_freq', 'chn_dev_click_freq',
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

cat_features = ['app', 'device','os', 'channel', 'hour']
target = 'is_attributed'
len_train = 60000000


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

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.07,
    'num_leaves': 20,  # 2^max_depth - 1
    'max_depth': 13,  # -1 means no limit
    'min_data_in_leaf': 20,
    'max_bin': 170,  # Number of bucketed bin for feature values
    'subsample': 0.05,  # Subsample ratio of the training instance.
    'colsample_bytree': 0.45,  # Subsample ratio of columns when constructing each tree.
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
                               num_boost_round=160,
                               early_stopping_rounds=0)

process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
del train_dataset, valid_dataset
gc.collect()
sys.stdout.flush()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = lgbm_model.predict(test_df[predictors])
if debug:
    sub.to_csv("./tmp/debug_sub.csv", index=False, float_format='%1.5f')
else:
    sub.to_csv("../output/lgbm_f26_SM_bot60m_gridCV_param.csv", index=False, float_format='%1.5f')

