"""
Use base model predictions as feature fore second level
training
"""

import gc
import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

debug = False
##################### load pre-processed data #####################
if debug:
    full_df = pd.read_pickle('./pre_proc_inputs/debug_f28_full_data.pkl')
    train_file = './pre_proc_inputs/debug_f28_lgbm_train.bin'
    train_len = 49999
    test_len = 49999
else:
    full_df = pd.read_pickle('./pre_proc_inputs/f28_full_data.pkl')
    train_file = './pre_proc_inputs/f28_lgbm_train.bin'
    train_len = 184903890
    test_len = 18790469


target = 'is_attributed'


sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = lgbm_model.predict(test_df[predictors])
if debug:
    sub.to_csv("./debug_sub.csv", index=False, float_format='%1.5f')
else:
    sub.to_csv("../output/lgbm_f22_SM_bot50m_gridCV_param.csv", index=False, float_format='%1.5f')
