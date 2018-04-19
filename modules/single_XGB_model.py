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

##################### if process data needed ######################
#train_file = '../input/train.csv'
#test_file = '../input/test.csv'
#full_df, len_train, predictors = read_merge_process(train_file, ftest=test_file)
#print("*************************  Full data info **********************************\n")
#full_df.info()
#print("*************************  End of data info **********************************\n")
#sys.stdout.flush()
            
#process = psutil.Process(os.getpid())
#print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
#sys.stdout.flush()

#print('all columns:\n', list(full_df), "\n")
#target = 'is_attributed'

#print('Predictors used for training: \n', predictors, "\n")
#cat_features = ['app', 'device','os', 'channel', 'weekday', 'hour']
#sys.stdout.flush()
##################### end of process data needed ######################

##################### load pre-processed data #####################
full_df = pd.read_pickle('./18_feature_bot60m_data.pkl')
predictors = ['app','device','os','channel','weekday','hour', 
                  'ip_app_count_chns', 'app_click_freq','app_freq',
                  'channel_freq',  'ip_day_hour_count_chns', 
                  'ip_app_os_count_chns',  'ip_day_chn_var_hour',
                  'ip_app_chn_mean_hour', 'ip_nextClick',  'ip_app_nextClick',
                  'ip_chn_nextClick', 'ip_os_nextClick' ]

cat_features = ['app', 'device','os', 'channel', 'weekday', 'hour']

len_train = 60000000
target = 'is_attributed'

print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
#################### end of loading processed data ###################################



print('total train data size: ', len_train)
#train_split = 170000000
train_split = 56000000
#train_split = 50000
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


Dtrain, Dvalid = XGB_DMatrix(train_df, valid_df, predictors, target)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del train_df, valid_df
gc.collect()

xgb_params = {'objective':'binary:logistic'
              , 'eval_metric' : 'auc'
              , 'tree_method':'approx'
              , 'gamma': 5.104e-8
              , 'learning_rate': 0.15
              , 'max_depth': 6
              , 'max_delta_step': 20
              , 'min_child_weight': 4
              , 'subsample': 1.0
              , 'colsample_bytree': 1.0
              , 'colsample_bylevel': 0.1
              , 'scale_pos_weight': 500
              , 'random_state': 300
              , 'reg_alpha': 1e-9
              , 'reg_lambda': 1000
              , 'silent': True
            }

xgb_model = single_XGB_train(  xgb_params, 
                               Dtrain, 
                               Dvalid, 
                               metrics='auc',
                               early_stopping_rounds=30 )

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del Dtrain, Dvalid
gc.collect()

print("Current model parameters:\n")
print(xgb_params)

Dtest = XGB_Dtest(test_df, predictors)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = xgb_model.predict(Dtest, ntree_limit=xgb_model.best_ntree_limit)
sub.to_csv("../output/xgb_f17_bot6m_SM_kernal_param.csv", index=False, float_format='%1.5f')

