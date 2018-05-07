"""
Model 1: LGBM model on full data
"""
import gc
import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd
import lightgbm as lgb

now = datetime.datetime.now()
print("\nModel id: 1")
print("Model type: LGBM\n")
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

debug = False
##################### load pre-processed data #####################
if debug:
    full_df = pd.read_pickle('./pre_proc_inputs/debug_f30_full_data.pkl')
    train_len = 49999
    test_len = 49999
else:
    full_df = pd.read_pickle('./pre_proc_inputs/f30_full_data.pkl')
    train_len = 184903890
    test_len = 18790469

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

print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
print("\nfeatures:\n", predictors)
sys.stdout.flush()
#################### end of loading processed data ###################################

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.07,
    'num_leaves': 20,  # 2^max_depth - 1
    'max_depth': 13,  # -1 means no limit
    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
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

early_stopping_rounds = 0
num_boost_round = 700
print("\n Model parameters:\n", lgbm_params)


# create cv indices

train_df = full_df[:train_len]
test_df = full_df[train_len:]
print('\ntrain size: ', len(train_df))
print('\ntest size: ', len(test_df))

# save out-of-sample prediction for train data and prediction for test data
test_pred = np.zeros((len(test_df),))

process = psutil.Process(os.getpid())
print("\n- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

fold_start = time.time()
train_set = lgb.Dataset(train_df[predictors].values, 
                        label=train_df[target].values,
                        feature_name=predictors,
                        categorical_feature=cat_features
                       )
evals_results = {}
print("No early stopping...\n")
bst = lgb.train(  lgbm_params
                , train_set
                , num_boost_round=num_boost_round
                , valid_sets=[train_set]
                , valid_names=['train']
                , evals_result=evals_results 
                , verbose_eval=10
               )
print("\nModel Report:\n")
print("AUC :", evals_results['train']['auc'][-1])
sys.stdout.flush()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = bst.predict(test_df[predictors])

print("\nEnd of full data training...")
