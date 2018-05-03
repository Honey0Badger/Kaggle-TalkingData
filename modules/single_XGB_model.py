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
full_df = pd.read_pickle('21_feature_bot60m_data.pkl')

predictors = [ 'app','channel','device','os','hour',
               'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 
               'X7', 'X8', 'app_click_freq', 'ip_tcount', 'ip_app_count', 
               'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
               'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
               'nextClick']
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
              , 'tree_method':'hist'
              , 'grow_policy':'lossguide'
              , 'gamma': 5.104e-8
              , 'learning_rate': 0.15
              , 'max_depth': 5
              , 'max_delta_step': 20
              , 'min_child_weight': 0.5
              , 'subsample': 0.4
              , 'colsample_bytree': 0.7
              , 'colsample_bylevel': 0.1
              , 'scale_pos_weight': 500
              , 'random_state': 300
              , 'reg_alpha': 1e-9
              , 'reg_lambda': 1000
              , 'silent': True
            }

print("Current model parameters:\n")
print(xgb_params)

xgb_model = single_XGB_train(  xgb_params, 
                               Dtrain, 
                               Dvalid, 
                               metrics='auc',
                               early_stopping_rounds=30 )

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del Dtrain, Dvalid
gc.collect()


Dtest = XGB_Dtest(test_df, predictors)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = xgb_model.predict(Dtest, ntree_limit=xgb_model.best_ntree_limit)
sub.to_csv("../output/xgb_f21_bot6m_SM_kernal_param.csv", index=False, float_format='%1.5f')

