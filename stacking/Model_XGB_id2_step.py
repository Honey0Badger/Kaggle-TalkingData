"""
Model 1: XGB on inputs with 28 features,
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
import xgboost as xgb


now = datetime.datetime.now()
print("\nModel id: 2")
print("Model type: XGB\n")
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
fold = 4

##################### load pre-processed data #####################
if debug:
    #full_df = pd.read_pickle('./pre_proc_inputs/debug_f28_full_data.pkl')
    train_file = './pre_proc_inputs/debug_f28_xgb_train.bin'
    test_file = './pre_proc_inputs/debug_f28_xgb_test.bin'
    train_len = 49999
    test_len = 49999
else:
    #full_df = pd.read_pickle('./pre_proc_inputs/f28_full_data.pkl')
    train_file = './pre_proc_inputs/f30_xgb_train.bin'
    test_file = './pre_proc_inputs/f30_xgb_test.bin'
    train_len = 184903890
    test_len = 18790469

predictors = ['nextClick', 'app','device','os', 'channel', 'hour', 
                  'app_click_freq', 'app_os_click_freq', 'app_dev_click_freq',
                  'chn_os_click_freq', 'chn_dev_click_freq',
                  'ip_tcount', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
                  'ip_app_nextClick','ip_chn_nextClick','ip_os_nextClick']
cat_features = ['app', 'device','os', 'channel', 'hour']

target = 'is_attributed'

#print("*************************  Full data info **********************************\n")
#full_df.info()
#print("*************************  End of data info **********************************\n")
print("\nfeatures:\n", predictors)
sys.stdout.flush()
#################### end of loading processed data ###################################

dtrain = xgb.DMatrix(train_file)
dtest = xgb.DMatrix(test_file)

params = {'objective':'binary:logistic'
              , 'eval_metric' : 'auc'
              , 'tree_method':'hist'
              , 'grow_policy':'lossguide'
              , 'gamma': 5.104e-8
              , 'learning_rate': 0.15
              , 'max_depth': 6
              , 'max_delta_step': 20
              , 'min_child_weight': 4
              , 'subsample': 1.0
              , 'colsample_bytree': 1.0
              , 'colsample_bylevel': 0.1
              , 'scale_pos_weight': 500
              , 'random_state': 300
              , 'reg_alpha': 1e-9
              , 'reg_lambda': 1000
              , 'silent': True
              , 'updater': 'grow_gpu'
            }

num_boost_round = 100
early_stopping_rounds = 40
print("\n Model parameters:\n", params)



print("Running %d fold cross validation...\n" % fold)
# save out-of-sample prediction for train data and prediction for test data
train_oos_pred = np.zeros((train_len,))
test_pred = np.zeros((test_len,))

# create cv indices
skf  = list(KFold(fold).split(train_oos_pred))
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
        train_set = dtrain.slice(train)
        val_set = dtrain.slice(val)
        evals_results = {}
        if early_stopping_rounds == 0:  # without early stopping
                print("No early stopping...\n")
                bst = xgb.train(params
                                , train_set
                                , num_boost_round=num_boost_round
                                , evals=[(train_set,'train'), (val_set, 'valid')]
                                , evals_result=evals_results 
                                , early_stopping_rounds=early_stopping_rounds
                                , verbose_eval=10
                                )
                print("\nModel Report:\n")
                print("AUC :", evals_results['valid']['auc'][-1])
                scores[i] = evals_results['valid']['auc'][-1]
                train_oos_pred[val] = bst.predict(val_set)
                test_pred = bst.predict(dtest)
                print("Fold %d fitting finished in %0.3fs" % (i, time.time() - fold_start))
        else:  # early stopping
                bst = xgb.train(params
                                , train_set
                                , num_boost_round=1000
                                , evals=[(train_set,'train'), (val_set, 'valid')]
                                , evals_result=evals_results 
                                , early_stopping_rounds=early_stopping_rounds
                                , verbose_eval=10
                                )
                best_round = bst.best_iteration
                print("\nModel Report:")
                print("N_estimators : ", best_round)
                print("AUC :", evals_results['valid']['auc'][best_round-1])
                scores[i] = evals_results['valid']['auc'][best_round-1]
                train_oos_pred[val] = bst.predict(val_set, ntree_limit=bst.best_ntree_limit)
                test_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
                print("Fold %d fitting finished in %0.3fs" % (i, time.time() - fold_start))

        print("Score for model is %f" % (scores[i]))
        np.savez(npzfile+'_'+str(i), fold=i, train_ind=val, train_pred=train_oos_pred[val]
                 , test_pred=test_pred, score=scores[i])
        process = psutil.Process(os.getpid())
        print("\n- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
        sys.stdout.flush()
        del train_set, val_set
        gc.collect()

print("\nEnd of current fold...")
