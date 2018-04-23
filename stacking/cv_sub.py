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
test_m1 = np.loadtxt('./model_outputs/test_model_id1_test_pred.csv', delimiter=',')
print("check dim of test from model 1: ", test_m1.shape)


sub['is_attributed'] = pd.Series(test_m1, index=sub.index)
sub.to_csv("./final_output/test_sub_LGBM_id1_small_train.csv", index=False, float_format='%1.5f')
