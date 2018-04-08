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
#train_data = load_from_file(train_file)
train_data = load_half_file(train_file)
train_size = train_data.shape
print("train data shape: ", train_size)
train_data.info()

train_df = train_DataFrame_processed(train_data)
sys.stdout.flush()
            
process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)

target = 'is_attributed'
predictors = ['ip','app', 'device','os', 'channel', 'weekday', 'hour']
cat_features = ['ip','app', 'device','os', 'channel', 'weekday', 'hour']

train_split = 80000
train = train_df[:train_split]
valid = train_df[train_split:]
train_dataset, valid_dataset = LGBM_DataSet(train, valid, predictors, target, cat_features)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del train_data, train_df, train, valid
gc.collect()

lgbm_params = {
    'learning_rate': 0.15,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'metric': 'auc',
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced 
}

lgbm_model = single_LGBM_train(lgbm_params, 
                               train_dataset, 
                               valid_dataset, 
                               metrics='auc',
                               early_stopping_rounds=20)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del train_dataset, valid_dataset
gc.collect()

# predict the test set
test_file = '../input/test_small.csv'
test_data = load_from_file(test_file)
Id = test_data['click_id'].values
test_size=test_data.shape
test_df = test_DataFrame_processed(test_data)
print ("feature extracted.")
print("test data size: ", test_size)
test_df.info()
sys.stdout.flush()

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)

submission = pd.DataFrame()
submission['is_attributed'] = lgbm_model.predict(test_df[predictors])
submission.insert(loc=0, column='click_id', value = Id)
submission.to_csv("../output/test_pred_lgbm.csv", index=False, float_format='%1.5f')

