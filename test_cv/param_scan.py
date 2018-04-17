import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd

from preprocess import *
from models import lgbm_est_base, xgb_est_base
from preprocess import *
from pipelines import *

start = time.time()

#train_file = '../input/train_CV_sample.csv'
#full_df, len_train, predictors = read_merge_process(train_file)

#full_df.to_pickle('18_feature.pkl')
full_df = pd.read_pickle('18_feature.pkl')
predictors = ['app','device','os','channel','weekday','hour', 
                  'ip_app_count_chns', 'app_click_freq','app_freq',
                  'channel_freq',  'ip_day_hour_count_chns', 
                  'ip_app_os_count_chns',  'ip_day_chn_var_hour',
                  'ip_app_chn_mean_hour', 'ip_nextClick',  'ip_app_nextClick',
                  'ip_chn_nextClick', 'ip_os_nextClick' ]
len_train = len(full_df)

print('**************** full data info *****************\n')
full_df.info()
print('**************** end of data info *****************\n')

target = 'is_attributed'

gridParams = {
         'learning_rate': [0.11],
         'n_estimators': [15]
         ,'max_depth': [7]
         ,'min_child_weight': [1e-3, 1e-2, 1e-1, 5e-1]
         ,'min_data_in_leaf': [30]
         ,'num_leaves': [40]
         ,'max_bin': [500]
         #,'random_state': [501]
         ,'colsample_bytree': [1.0]
         ,'subsample': [0.1]
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

search_model(full_df[predictors], full_df[target], xgb_est_base, gridParams_xgb, n_jobs=1, cv=10, refit=False)
