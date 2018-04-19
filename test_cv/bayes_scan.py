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

##################### load pre-processed data #####################
full_df = pd.read_pickle('../modules/18_feature_bot60m_data.pkl')

predictors = ['app','device','os','channel','weekday','hour', 
                  'ip_app_count_chns', 'app_click_freq','app_freq',
                  'channel_freq',  'ip_day_hour_count_chns', 
                  'ip_app_os_count_chns',  'ip_day_chn_var_hour',
                  'ip_app_chn_mean_hour', 'ip_nextClick',  'ip_app_nextClick',
                  'ip_chn_nextClick', 'ip_os_nextClick' ]

cat_features = ['app', 'device','os', 'channel', 'weekday', 'hour']

len_train = 20000000
full_df.drop(range(len_train,len(full_df.index)), inplace=True)

target = 'is_attributed'
print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
#################### end of loading processed data ###################################

#################### loading data from small samples #################################
#full_df = pd.read_pickle('18_feature.pkl')
#predictors = ['app','device','os','channel','weekday','hour', 
#                  'ip_app_count_chns', 'app_click_freq','app_freq',
#                  'channel_freq',  'ip_day_hour_count_chns', 
#                  'ip_app_os_count_chns',  'ip_day_chn_var_hour',
#                  'ip_app_chn_mean_hour', 'ip_nextClick',  'ip_app_nextClick',
#                  'ip_chn_nextClick', 'ip_os_nextClick' ]
#len_train = len(full_df)

#print('**************** full data info *****************\n')
#full_df.info()
#print('**************** end of data info *****************\n')

#target = 'is_attributed'
################### end of data loading ############################################


lgbm_param_space = {
                        'learning_rate': (0.01, 0.5, 'log-uniform')
                        ,'n_estimators': (50, 100)
                        ,'max_depth': (0, 50)
                        ,'min_child_weight': (0, 10)
                        ,'min_data_in_leaf': (20, 100)
                        ,'num_leaves': (10, 200)
                        ,'max_bin': (50, 500)
                        ,'colsample_bytree': (0.01, 1.0, 'uniform')
                        ,'subsample': (0.01, 1.0, 'uniform')
                        ,'scale_pos_weight': (0.1, 500, 'log-uniform')
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

Bayes_search(full_df[predictors], full_df[target], lgbm_est_base, lgbm_param_space, cv=kcv, iterations=100, n_jobs=3, refit=True)

print("\nfinished scanning at ", (time.time()-start)/3600)
