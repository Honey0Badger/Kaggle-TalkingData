"""
Model 1: LGBM on inputs with 28 features,
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
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

now = datetime.datetime.now()
print("\nModel id: 1")
print("Model type: LGBM\n")
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

debug = False
##################### load pre-processed data #####################
if debug:
    full_df = pd.read_pickle('./pre_proc_inputs/debug_f28_full_data.pkl')
    #train_file = './pre_proc_inputs/debug_f28_lgbm_train.bin'
    train_len = 49999
else:
    full_df = pd.read_pickle('../modules/22_feature_bot60m_data.pkl')
    #train_file = './pre_proc_inputs/f28_lgbm_train.bin'
    train_len = 60000000

predictors = [ 'app','channel','device','os','day','hour',
               'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 
               'X7', 'X8', 'ip_tcount', 'ip_app_count', 
               'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
               'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
               'nextClick', 'nextClick_shift' ]
cat_features = ['app', 'device','os', 'channel', 'day', 'hour']

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
    'learning_rate': 0.1,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 25,  # 2^max_depth - 1
    'max_depth': 12,  # -1 means no limit
    'min_child_samples': 70,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 400,  # Number of bucketed bin for feature values
    'subsample': 0.65,  # Subsample ratio of the training instance.
    #'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 1.0,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 1e-3,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight':300 # because training data is extremely unbalanced 
}
early_stopping_rounds = 0
num_boost_round = 60
print("\n Model parameters:\n", lgbm_params)


# create cv indices
fold = 10
if debug:
    holdout = 10000
else:
    holdout = 4000000

train_df = full_df[:train_len-holdout]
test_df = full_df[predictors][train_len-holdout:train_len]
holdout_y = full_df[target][train_len-holdout:train_len].values
skf  = list(KFold(fold).split(train_df))
print('\ntrain size: ', len(train_df))
print('\ntest size: ', len(test_df))

lgb_train_data = lgb.Dataset(train_df[predictors].values, label=train_df[target].values)

#lgb_train_data = lgb.Dataset(train_df[predictors].values, label=train_df[target].values, 
#                              feature_name=predictors)
# save out-of-sample prediction for train data and prediction for test data
train_oos_pred = np.zeros((len(train_df), 1))
test_pred = np.zeros((len(test_df), 1))
scores = np.zeros((fold,))

process = psutil.Process(os.getpid())
print("\n- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

for i, (train, val) in  enumerate(skf):
        print("Fold %d" % (i + 1))
        sys.stdout.flush()
        fold_start = time.time()
        train_set = lgb_train_data.subset(train)
        val_set = lgb_train_data.subset(val)
        #train_set = lgb_train_data
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
                #print("AUC :", evals_results['valid']['auc'][-1])
                #scores[i] = evals_results['valid']['auc'][-1]
                print("train_df: ", len(train_df))
                #train_oos_pred[val, 0] = bst.predict(train_df.iloc[val])
                fold_pred = bst.predict(test_df)
                check_pred = roc_auc_score(holdout_y, fold_pred)
                print("debug: check individual pred score: ", check_pred)
                test_pred[:, 0] = test_pred[:, 0] + fold_pred
                print("Fold %d fitting finished in %0.3fs" 
                        % (i + 1, time.time() - fold_start))
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
                print("train_df: ", len(train_df))
                #train_oos_pred[val, 0] = bst.predict(train_df.iloc[val])
                fold_pred = bst.predict(test_df)
                check_pred = roc_auc_score(holdout_y, fold_pred)
                test_pred[:, 0] = test_pred[:, 0] + fold_pred
                print("debug: check individual pred score: ", check_pred)
                print("Fold %d fitting finished in %0.3fs" 
                        % (i + 1, time.time() - fold_start))

        print("Score for model is %f" % (scores[i]))
        sys.stdout.flush()
        gc.collect()

test_pred = test_pred / fold
check_pred = roc_auc_score(holdout_y, test_pred.ravel())

#test_pred_blend = test_pred.mean(1)
print("compare mean test pred score %f" % check_pred)
print("Score for blended models is %f" % (np.mean(scores)))
sys.stdout.flush()

process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()


#if debug:
#    np.savetxt("./model_outputs/debug_model_id1_train.csv", train_oos_pred, fmt='%.5f', delimiter=",")
#    np.savetxt("./model_outputs/debug_model_id1_test.csv", test_pred_blend, fmt='%.5f', delimiter=",")
#else:
#    pass
#    np.savetxt("./model_outputs/model_id1_train_pred.csv", train_oos_pred, fmt='%.5f', delimiter=",")
#    np.savetxt("./model_outputs/model_id1_test_pred.csv", test_pred_blend, fmt='%.5f', delimiter=",")

