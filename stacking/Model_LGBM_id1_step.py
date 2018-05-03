"""
Model 1: LGBM model, one cv step
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
from sklearn.model_selection import KFold
import lightgbm as lgb

now = datetime.datetime.now()
print("\nModel id: 1")
print("Model type: LGBM\n")
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
if len(sys.argv) <= 2:
    print("Usage: python *.py npzfile [fold #]")
    exit()
else:
#    print("debug:", sys.argv)
    npzfile = sys.argv[1]
    run_folds = list(map(int, sys.argv[2:]))
    print("Run the following folds for Model...", run_folds)
    print("saving temperory data to file...", npzfile)
sys.stdout.flush()

start = time.time()

debug = False
fold = 10
##################### load pre-processed data #####################
if debug:
    full_df = pd.read_pickle('./pre_proc_inputs/debug_f21_full_data.pkl')
    train_len = 49999
    test_len = 49999
else:
    full_df = pd.read_pickle('./pre_proc_inputs/f21_full_data.pkl')
    train_len = 184903890
    test_len = 18790469

predictors = [ 'app','channel','device','os','hour',
               'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 
               'X7', 'X8', 'app_click_freq', 'ip_tcount', 'ip_app_count', 
               'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
               'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
               'nextClick']
cat_features = ['app', 'device','os', 'channel', 'hour']

target = 'is_attributed'

print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
print("\nfeatures:\n", predictors)
sys.stdout.flush()
#################### end of loading processed data ###################################

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.07,
    'num_leaves': 20,  # 2^max_depth - 1
    'max_depth': 13,  # -1 means no limit
    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 170,  # Number of bucketed bin for feature values
    'subsample': 0.05,  # Subsample ratio of the training instance.
    'colsample_bytree': 0.45,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 1e-3,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight':300 # because training data is extremely unbalanced 
}
early_stopping_rounds = 30
num_boost_round = 100
print("\n Model parameters:\n", lgbm_params)


# create cv indices

train_df = full_df[:train_len]
test_df = full_df[train_len:]
skf  = list(KFold(fold).split(train_df))
print('\ntrain size: ', len(train_df))
print('\ntest size: ', len(test_df))

# save out-of-sample prediction for train data and prediction for test data
print("Running %d fold cross validation...\n" % fold)
train_oos_pred = np.zeros((len(train_df),))
test_pred = np.zeros((len(test_df),))
scores = np.zeros((fold,))

process = psutil.Process(os.getpid())
print("\n- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

for i, (train, val) in  enumerate(skf):
        if i not in run_folds:
            continue
        print("Fold %d" % (i))
        sys.stdout.flush()
        fold_start = time.time()
        train_set = lgb.Dataset(train_df[predictors].iloc[train].values, 
                               label=train_df[target].iloc[train].values,
                               feature_name=predictors,
                               categorical_feature=cat_features
                               )
        val_set = lgb.Dataset(train_df[predictors].iloc[val].values, 
                               label=train_df[target].iloc[val].values,
                               feature_name=predictors,
                               categorical_feature=cat_features
                               )
        evals_results = {}
        if early_stopping_rounds == 0:  # without early stopping
                print("No early stopping...\n")
                bst = lgb.train(  lgbm_params
                                , train_set
                                , num_boost_round=num_boost_round
                                , valid_sets=[train_set, val_set]
                                , valid_names=['train', 'valid']
                                , evals_result=evals_results 
                                , verbose_eval=10
                               )
                print("\nModel Report:\n")
                print("AUC :", evals_results['valid']['auc'][-1])
                scores[i] = evals_results['valid']['auc'][-1]
                train_oos_pred[val] = bst.predict(train_df[predictors].iloc[val])
                test_pred = bst.predict(test_df[predictors])
                print("Fold %d fitting finished in %0.3fs" % (i, time.time() - fold_start))
        else:  # early stopping
                print("use early stopping...\n")
                bst = lgb.train(  lgbm_params
                                , train_set
                                , num_boost_round=1000
                                , valid_sets=[train_set, val_set]
                                , valid_names=['train', 'valid']
                                , evals_result=evals_results 
                                , early_stopping_rounds=early_stopping_rounds
                                , verbose_eval=10
                               )
                best_round = bst.best_iteration
                print("\nModel Report:\n")
                print("best round : ", best_round)
                print("AUC :", evals_results['valid']['auc'][best_round-1])
                scores[i] = evals_results['valid']['auc'][best_round-1]
                train_oos_pred[val] = bst.predict(train_df[predictors].iloc[val])
                test_pred = bst.predict(test_df[predictors])
                print("Fold %d fitting finished in %0.3fs" % (i, time.time() - fold_start))

        print("Score for model is %f" % (scores[i]))
        np.savez(npzfile+'_'+str(i), fold=i, train_ind=val, train_pred=train_oos_pred[val]
                 , test_pred=test_pred, score=scores[i])
        sys.stdout.flush()
        del train_set, val_set
        gc.collect()

print("\nEnd of current fold...")
