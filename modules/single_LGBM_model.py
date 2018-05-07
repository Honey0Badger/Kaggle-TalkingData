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


def add_feature(full_df, feature):
    fname = './features/'+feature+'.csv'
    f_col = pd.read_csv(fname)
    if len(f_col) != len(full_df):
        print("feature %s dimension not equal to data dim" % (feature))
        exit()
    full_df[feature] = f_col[feature].values
    del f_col
    gc.collect()
    return full_df


now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

debug = False

##################### load pre-processed data #####################
full_df = pd.read_pickle('28_feature_bot60m_data.pkl')

predictors = ['nextClick', 'app','device','os', 'channel', 'hour', 
                  'app_click_freq', 'app_os_click_freq', 'app_dev_click_freq',
                  'chn_os_click_freq', 'chn_dev_click_freq',
                  'ip_tcount', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
                  'ip_app_nextClick','ip_chn_nextClick','ip_os_nextClick']
cat_features = ['app', 'device','os', 'channel', 'hour']
target = 'is_attributed'

# adding features from separate files
#full_df = add_feature( full_df, 'ip_app_nextClick')
#predictors.append('ip_app_nextClick')
#full_df = add_feature('ip_os_next_Click')
#full_df = add_feature('ip_chn_nextClick')
#full_df = add_feature('ip_app_nextClick')

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
    'learning_rate': 0.1,
    'num_leaves': 20,  # 2^max_depth - 1
    'max_depth': 6,  # -1 means no limit
    'min_data_in_leaf': 90,
    'max_bin': 150,  # Number of bucketed bin for feature values
    'subsample': 0.9,  # Subsample ratio of the training instance.
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
                               num_boost_round=1000,
                               early_stopping_rounds=50)

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
    sub.to_csv("../output/lgbm_f30_SM_bot60m_gridCV_param.csv", index=False, float_format='%1.5f')

