import lightgbm as lgb
import numpy as np 
import pandas as pd 


## for LightGBM models
lgb_param1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 20,  
        'max_depth': 6,  
        'min_data_in_leaf': 90,
        'max_bin': 150,  
        'subsample': 0.9,
        'colsample_bytree': 1.0, 
        'min_child_weight': 1e-3,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'scale_pos_weight':300 # because training data is extremely unbalanced 
        }

lgb_param2 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 20,  
        'max_depth': 6,  
        'min_data_in_leaf': 90,
        'max_bin': 150,  
        'subsample': 0.9,
        'colsample_bytree': 1.0, 
        'min_child_weight': 1e-3,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'scale_pos_weight':300 # because training data is extremely unbalanced 
        }
