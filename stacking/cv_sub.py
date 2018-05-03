"""
This script directly creates submission from cv results
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

# read original train DataFrame as target
test_df = pd.read_csv('../input/test.csv', parse_dates=['click_time'])
print("reference data dimensions:\n")
print("test size: ", len(test_df))

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

del test_df
gc.collect()


# read all test predictions
test_m1 = np.loadtxt('./model_outputs/f21_xgb_4fold_cv_test.csv', delimiter=',')
print("check dim of test from model 1: ", test_m1.shape)

# simgple average 

test_m2 = np.loadtxt('./model_outputs/f21_lgbm_4fold_cv_test.csv', delimiter=',')
print("check dim of test from model 2: ", test_m2.shape)

w1 = 0.4
test_avg = test_m1 * w1 + test_m2 * (1-w1)

sub['is_attributed'] = pd.Series(test_avg, index=sub.index)
sub.to_csv("./final_output/sub_f21_LGBM6_XGB4_mean_pred.csv", index=False, float_format='%1.5f')
