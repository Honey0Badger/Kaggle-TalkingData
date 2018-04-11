import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd

from preprocess import *
from models import *
from pipelines import *

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

train_file = '../input/train.csv'
test_file = '../input/test.csv'
full_df, len_train = read_merge_process(train_file, ftest=test_file)
print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
sys.stdout.flush()
            
process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

print('all columns:\n', list(full_df), "\n")
target = 'is_attributed'
predictors = ['app','device','os','channel','weekday','hour', 
              'ip_app_count_chns', 'app_click_freq','app_freq',
               'channel_freq']

print('Predictors used for training: \n', predictors, "\n")
cat_features = ['app', 'device','os', 'channel', 'weekday', 'hour']
sys.stdout.flush()

train_split = 46000000
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
    'learning_rate': 0.05,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 30,  # 2^max_depth - 1
    'max_depth': 10,  # -1 means no limit
    'min_child_samples': 1000,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.2,  # Subsample ratio of the training instance.
    'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
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
                               early_stopping_rounds=20)

process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
del train_dataset, valid_dataset
gc.collect()
sys.stdout.flush()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = lgbm_model.predict(test_df[predictors])
sub.to_csv("../output/test_pred_lgbm_new_feature.csv", index=False, float_format='%1.5f')

