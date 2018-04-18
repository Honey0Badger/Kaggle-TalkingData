import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import StratifiedKFold, KFold
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

xgb_param_space = {
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'min_child_weight': (0, 10),
                'max_depth': (0, 50),
                'max_delta_step': (0, 20),
                'subsample': (0.01, 1.0, 'uniform'),
                'colsample_bytree': (0.01, 1.0, 'uniform'),
                'colsample_bylevel': (0.01, 1.0, 'uniform'),
                'reg_lambda': (1e-9, 1000, 'log-uniform'),
                'reg_alpha': (1e-9, 1.0, 'log-uniform'),
                'gamma': (1e-9, 0.5, 'log-uniform'),
                'min_child_weight': (0, 5),
                'n_estimators': (50, 100),
                'scale_pos_weight': (1e-6, 500, 'log-uniform')
         }

kcv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

Bayes_search(full_df[predictors], full_df[target], xgb_est_base2, xgb_param_space, cv=kcv, iterations=10, n_jobs=3, refit=True)

print("\nfinished scanning at ", (time.time()-start)/3600)
