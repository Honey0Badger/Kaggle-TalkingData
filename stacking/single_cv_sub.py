"""
This script reconstruct the data from model*_step runs
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


# specify the overall parameters for the model
# check those parameters carefully before each run
prefix = './tmp_data/lgbm_cv'
outfile = './final_output/f24_lgbm_4fold'
fold = 3

# reconstruct train and test predictions
npzfile=prefix + '_' + str(fold) + '.npz'
print('load from fold %d file...' % fold)
dat=np.load(npzfile)
if fold != dat['fold']:
    print('Error: specified fold and data ind not consistent!\n')
    exit()
test_pred = dat['test_pred']
print('Scores for each fold: ', dat['score'])

test_df = pd.read_csv('../input/test.csv', parse_dates=['click_time'])
print("reference data dimensions:\n")
print("test size: ", len(test_df))

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

del test_df
gc.collect()

outfile=outfile + '_' + str(fold) + '.csv'
sub['is_attributed'] = pd.Series(test_pred, index=sub.index)
sub.to_csv(outfile, index=False, float_format='%1.5f')
