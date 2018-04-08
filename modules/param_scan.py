import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd

from preprocess import *
from models import lgbm_est_base
from preprocess import *
from pipelines import *

start = time.time()

train_file = '../input/train_CV_sample.csv'


#train_data = load_from_file(train_file)
train_data = load_half_file(train_file)
train_size = train_data.shape

train_x, train_y = process_trainData(train_data)
print("train data size: ", train_size)

del train_data
gc.collect()

gridParams = {
         #'learning_rate': [0.05]
         'n_estimators': [16, 18, 20, 22, 24, 26]
         #,'max_depth': [15]
         #,'min_child_weight': [1e-3, 5e-3, 1e-2, 5e-2]
         #,'min_data_in_leaf': [1800]
         #,'num_leaves': [70]
         #,'random_state': [501]
         #,'colsample_bytree': [0.3]
         #,'subsample': [0.6]
         #,'is_unbalance': [True]
         }

gridParams_xgb = {
         #'learning_rate': [0.1]
          'n_estimators': [10, 20, 30, 40, 60]
         # ,'max_depth': [3]
         #,'min_child_weight': [1e-3]
         #,'min_data_in_leaf': [1800]
         #,'num_leaves': [70]
         #,'random_state': [501]
         #,'colsample_bytree': [0.3]
         #,'is_unbalance': [True]
         }
# view default model params
#print("Default parameters:")
#print(lgbm_est_base.get_params())

search_model(train_x, train_y, est_XGB_class[0], gridParams_xgb, n_jobs=-1, cv=4, refit=False)


