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


## read all test predictions
#test_m1 = np.loadtxt('./model_outputs/f21_lgbm_4fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 1: ", test_m1.shape)
#
#test_m2 = np.loadtxt('./model_outputs/f21_xgb_4fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 2: ", test_m2.shape)
#
#test_m3 = np.loadtxt('./model_outputs/f21_lgbm_10fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 3: ", test_m3.shape)
#
#test_m4 = np.loadtxt('./model_outputs/f24_lgbm_4fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 4: ", test_m4.shape)
#
#test_m5 = np.loadtxt('./model_outputs/f24_xgb_4fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 5: ", test_m5.shape)
#
#test_m6 = np.loadtxt('./model_outputs/f30_lgbm_4fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 6: ", test_m6.shape)
#
#test_m7 = np.loadtxt('./model_outputs/f30_lgbm_4fold_gridCV_test.csv', delimiter=',')
#print("check dim of test from model 7: ", test_m7.shape)
#
#test_m8 = np.loadtxt('./model_outputs/f30_xgb_4fold_cv_test.csv', delimiter=',')
#print("check dim of test from model 8: ", test_m8.shape)
##sub['is_attributed'] = pd.Series(test_m2, index=sub.index)
##sub.to_csv("./final_output/sub_f30_LGBM_4fold_gridCV_pred.csv", index=False, float_format='%1.5f')
#
#w1 = 0.12
#w2 = 0.16
#test_avg = (test_m1 + test_m2 + test_m4 + test_m5 \
#            + test_m6 + test_m7 + test_m8) * w1 + test_m3 * w2

test_m1 = pd.read_csv('./model_outputs/sub-it200102.csv')

test_m2 = pd.read_csv('./final_output/sub_all_model_mean_pred.csv')

w1 = 0.6
w2 = 0.4
test_avg1 = test_m1['is_attributed']* w1 + test_m2['is_attributed'] * w2
sub['is_attributed'] = pd.Series(test_avg1, index=sub.index)
sub.to_csv("./final_output/sub_allmodel4_one6_pred.csv", index=False, float_format='%1.5f')
del test_avg1
gc.collect()

w1 = 0.4
w2 = 0.6
test_avg2 = test_m1['is_attributed']* w1 + test_m2['is_attributed'] * w2
sub['is_attributed'] = pd.Series(test_avg2, index=sub.index)
sub.to_csv("./final_output/sub_allmodel6_one4_pred.csv", index=False, float_format='%1.5f')
del test_avg2
gc.collect()


w1 = 0.45
w2 = 0.55
test_avg3 = test_m1['is_attributed']* w1 + test_m2['is_attributed'] * w2
sub['is_attributed'] = pd.Series(test_avg3, index=sub.index)
sub.to_csv("./final_output/sub_allmodel55_one45_pred.csv", index=False, float_format='%1.5f')
del test_avg3
gc.collect()

w1 = 0.55
w2 = 0.45
test_avg4 = test_m1['is_attributed']* w1 + test_m2['is_attributed'] * w2
sub['is_attributed'] = pd.Series(test_avg4, index=sub.index)
sub.to_csv("./final_output/sub_allmodel45_one55_pred.csv", index=False, float_format='%1.5f')
del test_avg4
gc.collect()
