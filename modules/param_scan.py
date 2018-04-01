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

train_file = '../input/train_Attributed_Sample.csv'


train_data = load_from_file(train_file)
train_size = train_data.shape

train_x, train_y = process_trainData(train_data)
print("train data size: ", train_size)

del train_data
gc.collect()

gridParams = {
         'learning_rate': [0.01]
         #,'n_estimators': [8, 16, 24]
         ,'num_leaves': [32]
         ,'boosting_type': ['gbdt']
         ,'objective': ['binary']
         ,'random_state': [501]
         ,'colsample_bytree': [0.3, 0.32]
         ,'subsample': [0.7, 0.75]
         ,'reg_alpha': [0.0]
         ,'reg_lambda': [0.0]
         ,'is_unbalance': [True]
         }

# view default model params
print("Default parameters:")
print(lgbm_est_base.get_params())

search_model(train_x, train_y, lgbm_est_base, gridParams, n_jobs=-1, cv=2, refit=False)


