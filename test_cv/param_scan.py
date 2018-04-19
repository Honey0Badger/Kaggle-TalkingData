import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd

#from preprocess import *
from models import lgbm_est_base, xgb_est_base
from preprocess2 import *
from pipelines import *

start = time.time()

#train_file = '../input/train_CV_sample.csv'
#full_df, len_train, predictors = read_merge_process2(train_file)
#full_df.to_pickle('22_feature.pkl')

full_df = pd.read_pickle('22_feature.pkl')
predictors = [ 'app','channel','device','os','day','hour',
               'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 
               'X7', 'X8', 'ip_tcount', 'ip_app_count', 
               'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
               'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
               'nextClick', 'nextClick_shift' ]
len_train = len(full_df)

print('**************** full data info *****************\n')
full_df.info()
print('**************** end of data info *****************\n')

target = 'is_attributed'

gridParams = {
         'learning_rate': [0.1],
         'n_estimators':  [30]
         ,'max_depth': [12]
         ,'min_child_weight': [1e-3]
         ,'min_data_in_leaf': [70]
         ,'num_leaves': [25]
         ,'max_bin':  [400]
         ,'random_state': [501]
         ,'colsample_bytree': [1.0]
         ,'subsample': [0.65]
         #,'is_unbalance': [True]
         }

gridParams_xgb = {
          'learning_rate': [0.15]
          ,'n_estimators': [25]
          ,'max_depth': [3]
          ,'min_child_weight': [1e-3]
          , 'max_leaves': [500, 1000, 1500, 2000]
          #,'min_data_in_leaf': [1800]
          #,'max_leaf_nodes': [20, 40, 50, 70, 90]
         #,'random_state': [501]
          ,'colsample_bytree': [0.5]
          #,'colsample_bylevel': [0.0, 0.4, 0.6, 0.8, 1.0]
          ,'subsample': [1.0]
         #,'is_unbalance': [True]
         }
# view default model params
#print("Default parameters:")
#print(lgbm_est_base.get_params())

search_model(full_df[predictors], full_df[target], lgbm_est_base, gridParams, n_jobs=1, cv=10, refit=False)
