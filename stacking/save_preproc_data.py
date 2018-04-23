import os
import gc
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd
import lightgbm as lgb
import xgboost as xgb

#from preprocess import *
from preprocess2 import *

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

train_file = '../input/train_debug.csv'
test_file = '../input/test_debug.csv'
full_df, len_train, predictors = read_merge_process2(train_file, ftest=test_file)
cat_features = ['app', 'device','os', 'channel', 'day', 'hour']
target = 'is_attributed'
print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
sys.stdout.flush()
            
process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

#full_df.to_pickle('./pre_proc_inputs/debug_f28_full_data.pkl')

train_df = full_df[:len_train]
test_df = full_df[len_train:]

# lgbm data for train
print('\nconverting train to lgbm format...')
lgbm_DS = lgb.Dataset( train_df[predictors].values, label=train_df[target].values,
                        feature_name=predictors,
                        categorical_feature=cat_features
                        )
lgbm_DS.save_binary('./pre_proc_inputs/debug_f28_lgbm_train.bin')
del lgbm_DS
gc.collect()
process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)

## debug ##
lgbm_reload = lgb.Dataset('./pre_proc_inputs/debug_f28_lgbm_train.bin', feature_name=predictors, categorical_feature=cat_features)
print('cat features: ', lgbm_reload.categorical_feature)
###########

## lgbm data for test (unnecessary for lgbm)
#print('\nconverting test to lgbm format...')
#lgbm_DS = lgb.Dataset( test_df[predictors].values, label=test_df[target].values,
#                        feature_name=predictors,
#                        categorical_feature=cat_features
#                        )
#lgbm_DS.save_binary('./pre_proc_inputs/f28_lgbm_test.bin')
#del lgbm_DS
#gc.collect()
#process = psutil.Process(os.getpid())
#print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)

## xgb data for train
#print('\nconverting test to xgb format...')
#dtrain = xgb.DMatrix( train_df[predictors], label=train_df[target].values )
#dtrain.save_binary('./pre_proc_inputs/debug_f28_xgb_train.bin')
#del dtrain
#gc.collect()
#process = psutil.Process(os.getpid())
#print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
#
## xgb data for test
#print('\nconverting test to xgb format...')
#dtest = xgb.DMatrix( test_df[predictors], label=test_df[target].values )
#dtest.save_binary('./pre_proc_inputs/debug_f28_xgb_test.bin')
#del dtest
#gc.collect()
#
