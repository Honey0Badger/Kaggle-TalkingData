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

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet,Ridge,LinearRegression
import xgboost as xgb
import pandas as pd

from preprocess2 import *

def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False, save_log=None):
    ##Grid Search for the best model
    model = GridSearchCV(estimator=est,
                         param_grid=param_grid,
                         scoring='roc_auc',
                         verbose=10,
                         n_jobs=n_jobs,
                         iid=True,
                         refit=refit,
                         cv=cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("mean test scores:\n")
    print(model.cv_results_.__getitem__('mean_test_score'))
    print("std test scores:\n")
    print(model.cv_results_.__getitem__('std_test_score'))
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("**********************************************")
    sys.stdout.flush()

    if save_log != None:
        # saving all model scores
        np.savez(save_log, model.cv_results_)

    return model


# read original train DataFrame as target
train_df = load_from_file('../input/train_debug.csv')
test_df = load_from_file('../input/test_debug.csv')
print("reference data dimensions:\n")
print("train size: ", len(train_df))
print("test size: ", len(test_df))

train_label = train_df.is_attributed.astype('int8').values
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

del train_df, test_df
gc.collect()

# read all out-of-sample train predictions
train_m1 = np.loadtxt('./model_outputs/debug_model_id1_train.csv', delimiter=',')
print("check dim of train from model 1: ", train_m1.shape)

train_m2 = np.loadtxt('./model_outputs/debug_model_id2_train.csv', delimiter=',')
print("check dim of train from model 2: ", train_m2.shape)

train_m3 = np.loadtxt('./model_outputs/debug_model_id3_train.csv', delimiter=',')
print("check dim of train from model 3: ", train_m3.shape)

# read all test predictions
test_m1 = np.loadtxt('./model_outputs/debug_model_id1_test.csv', delimiter=',')
print("check dim of test from model 1: ", train_m1.shape)

test_m2 = np.loadtxt('./model_outputs/debug_model_id2_test.csv', delimiter=',')
print("check dim of test from model 2: ", train_m2.shape)

test_m3 = np.loadtxt('./model_outputs/debug_model_id3_test.csv', delimiter=',')
print("check dim of test from model 3: ", train_m3.shape)

# linear blending seems to give negative predictions, might not be suitable for this problem
print  ("\nLinear Blending.")
param_grid = {
    'alpha':[0, 0.0001, 70]
              }
model = search_model(np.hstack((train_m1.reshape((-1,1)),
                                train_m2.reshape((-1,1)),
                                train_m3.reshape((-1,1))
                                ))
                                , train_label
                                , Ridge()
                                , param_grid
                                , n_jobs=1
                                , cv=2
                                , refit=True
                                , save_log='./final_output/linear_blend.npz'
                    )

pred_y_ridge = model.predict(np.hstack((test_m1.reshape((-1,1)),
                                        test_m2.reshape((-1,1)),
                                        test_m3.reshape((-1,1))
                                      ))
                            )
                                              

# for testing
#test_blend = test_m1

sub['is_attributed'] = pd.Series(pred_y_ridge, index=sub.index)
sub.to_csv("./final_output/test_sub.csv", index=False, float_format='%1.5f')

print("\nXGBoost blending")
params = {
        'objective':'binary:logistic',
        'tree_method':'hist',
        'grow_policy':'lossguide',
        'eta': 0.01,
        'eval_metric' : 'auc',
        'lambda': 0,
        'alpha': 0, # you can try different values for alpha
        'lambda_bias' : 0,
        'silent': 0,
        'seed': 1234
}


xgtrain_blend = xgb.DMatrix(np.hstack((train_m1.reshape((-1,1)),
                                       train_m2.reshape((-1,1)),
                                       train_m3.reshape((-1,1))
                                      )),
                            label=train_label,
                            missing=np.nan)

xgtest_blend = xgb.DMatrix(np.hstack((test_m1.reshape((-1,1)),
                                      test_m2.reshape((-1,1)),
                                      test_m3.reshape((-1,1))
                                      )))

xgb_model=xgb.train(params, xgtrain_blend, evals=[(xgtrain_blend, 'train')], num_boost_round=100, verbose_eval=10)
pred_xgb = xgb_model.predict(xgtest_blend)

sub['is_attributed'] = pd.Series(pred_y_ridge, index=sub.index)
sub.to_csv("./final_output/test_sub_xgb.csv", index=False, float_format='%1.5f')
