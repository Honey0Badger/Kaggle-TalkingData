import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd

from models import lgbm_est_base, xgb_est_base
from preprocess4 import *
from pipelines import *

start = time.time()

full_df = pd.read_csv('f30_cv_sample.csv')

predictors = ['nextClick', 'app','device','os', 'channel', 'hour', 
                  'app_click_freq', 'app_os_click_freq', 'app_dev_click_freq',
                  'chn_os_click_freq', 'chn_dev_click_freq',
                  'ip_tcount', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
                  'ip_app_nextClick','ip_chn_nextClick','ip_os_nextClick']
target = 'is_attributed'

len_train = len(full_df)
print('**************** full data info *****************\n')
full_df[predictors+[target]].info()
print('**************** end of data info *****************\n')


gridParams_lgbm = {
         'learning_rate': [0.1],
         'n_estimators':  [170]
         ,'max_depth': [6]
         ,'min_child_weight': [1e-3]
         ,'min_data_in_leaf': [90]
         ,'num_leaves': [20]
         ,'max_bin':  [150]
         ,'random_state': [501]
         ,'colsample_bytree': [1.0]
         ,'subsample': [0.9]
         #,'is_unbalance': [True]
         }

gridParams_xgb = {
                'gamma': [5.104e-8]
              , 'learning_rate': [0.15]
              , 'n_estimators': [25]
              , 'max_depth': [5]
              , 'max_delta_step': [20]
              , 'min_child_weight': [0.5]
              , 'subsample': [0.4]
              , 'colsample_bytree': [0.7]
              , 'colsample_bylevel': [0.1]
              , 'reg_alpha': [1e-9]
              , 'reg_lambda': [1000]
            }
#gridParams_xgb = {
#          'learning_rate': [0.2]
#          ,'n_estimators': [18]
#          , 'gamma': [1e-8, 5.104e-8, 1e-9]
#          ,'max_depth': [3]
#          , 'max_delta_step': [20]
          #,'min_child_weight': [1e-3]
          #,'min_data_in_leaf': [50, 100, 200, 500]
         #,'random_state': [501]
          #,'colsample_bytree': [0.5]
          #,'colsample_bylevel': [0.0, 0.4, 0.6, 0.8, 1.0]
          #,'subsample': [1.0]
         #,'is_unbalance': [True]
#         }
# view default model params
#print("Default parameters:")
#print(lgbm_est_base.get_params())

search_model(full_df[predictors], full_df[target], lgbm_est_base, gridParams_lgbm, n_jobs=1, cv=5, refit=False)
