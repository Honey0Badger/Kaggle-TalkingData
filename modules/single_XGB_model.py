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
features = ['ip','app', 'device','os', 'channel', 'weekday', 'hour']

train_split = 80000
train = train_df[:train_split]
valid = train_df[train_split:]
Dtrain, Dvalid = XGB_DMatrix(train, valid, features, target)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del train_data, train_df, train, valid
gc.collect()

xgb_params = {'objective':'binary:logistic'
              , 'eta':0.3
              , 'tree_method':'hist'
              , 'grow_policy':'lossguide'
              , 'max_leaves': 1400
              , 'max_depth': 0
              , 'subsample': 0.9
              , 'colsample_bytree': 0.7
              , 'colsample_bylevel': 0.7
              , 'min_child_weight':0
              , 'alpha': 4
              , 'scale_pos_weight': 300
              , 'eval_metric': 'auc'
              , 'random_state': 300
              , 'silent': True
              #, n_estimators=10
            }

xgb_model = single_XGB_train(  xgb_params, 
                               Dtrain, 
                               Dvalid, 
                               metrics='auc',
                               early_stopping_rounds=20)

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)
del Dtrain, Dvalid
gc.collect()

# predict the test set
test_file = '../input/test_small.csv'
test_data = load_from_file(test_file)
Id = test_data['click_id'].values
test_size=test_data.shape
test_df = test_DataFrame_processed(test_data)
Dtest = XGB_Dtest(test_df, features)
print ("feature extracted.")
print("test data size: ", test_size)
test_df.info()
sys.stdout.flush()

process = psutil.Process(os.getpid())
print("Current process memory usage: ", process.memory_info().rss/1048576)

submission = pd.DataFrame()
submission['is_attributed'] = xgb_model.predict(Dtest, ntree_limit=xgb_model.best_ntree_limit)
submission.insert(loc=0, column='click_id', value = Id)
submission.to_csv("../output/test_pred_xgb.csv", index=False, float_format='%1.5f')

