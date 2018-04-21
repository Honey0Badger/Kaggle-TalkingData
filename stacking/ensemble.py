"""
This script takes varial first-level model outputs 
and ensemble them
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

from sklearn.linear_model import ElasticNet,Ridge,LinearRegression
import pandas as pd

from preprocess2 import *
from pipelines import search_model

# read original train DataFrame as target
train_df = load_from_file('../input/train_debug.csv')
test_df = load_from_file('../input/test_debug.csv')
print("reference data dimensions:\n")
print("train size: ", len(train_df))
print("test size: ", len(test_df))

train_label = train_df.is_attributed.values
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

del train_df, test_df
gc.collect()

# read all out-of-sample train predictions
train_m1 = np.loadtxt('./model_outputs/debug_model_id1_train.csv', delimiter=',')
print("check dim of train from model 1: ", train_m1.shape)

train_m2 = np.loadtxt('./model_outputs/debug_model_id2_train.csv', delimiter=',')
print("check dim of train from model 2: ", train_m2.shape)

# read all test predictions
test_m1 = np.loadtxt('./model_outputs/debug_model_id1_test.csv', delimiter=',')
print("check dim of test from model 1: ", train_m1.shape)

test_m2 = np.loadtxt('./model_outputs/debug_model_id2_test.csv', delimiter=',')
print("check dim of test from model 2: ", train_m2.shape)


print  ("\nLinear Blending.")
param_grid = {
    'alpha':[0, 0.0001, 70]
              }
model = search_model(np.hstack((train_m1.reshape((-1,1)),
                                train_m2.reshape((-1,1)),
                                ))
                                , train_y
                                , Ridge()
                                , param_grid
                                , n_jobs=1
                                , cv=2
                                , refit=True
                                , save_log='./final_output/linear_blend.npz'
                    )

pred_y_ridge = model.predict(np.hstack((test_m1.reshape((-1,1)),
                                        test_m2.reshape((-1,1))
                                      ))
                            )
                                              

# for testing
test_blend = test_m1

sub['is_attributed'] = pd.Series(test_blend, index=sub.index)
sub.to_csv("./final_output/test_sub.csv", index=False, float_format='%1.5f')


