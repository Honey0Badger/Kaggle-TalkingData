import xgboost as xgb
import lightgbm as lgb


from metric import *


estimators_LGBM = [lgb.LGBMClassifier(   learning_rate=0.01
                                       , boosting_type='gbdt'
                                       , objective='binary'
                                       , num_leaves=31
                                       , max_depth=-1
                                       , max_bin=255
                                       , min_child_samples=20    # min_child_samples in lgb
                                       , subsample=0.6
                                       , subsample_freq=0
                                       , colsample_bytree=0.3
                                       , min_child_weight=5
                                       , subsample_for_bin=200000
                                       , is_unbalance=True
                                       , verbose = 0 )
#                   , lgb.LGBMClassifier( learning_rate=0.01,
#                                      objective='binary',
#                                       is_unbalance=True,
#                                       verbose = 0 )
                   ]

est_LGBM_test = [lgb.LGBMClassifier(   learning_rate=0.25
                                       , boosting_type='gbdt'
                                       , objective='binary'
                                       , n_estimators=18
                                       , num_leaves=70
                                       , max_depth=15
                                       , subsample=0.6
                                       , colsample_bytree=0.3
                                       , min_child_weight=1e-3
                                       , min_data_in_leaf=1800
                                       , is_unbalance=True
                                       , verbose = 0 )
                   ]

lgbm_est_base = lgb.LGBMClassifier(   learning_rate=0.1
                                       , boosting_type='gbdt'
                                       , objective='binary'
                                       , is_unbalance=True
                                       #, scale_pos_weight =300 # because training data is extremely unbalanced 
                                       , verbose = 0 )

lgbm_f10 = {lgb.LGBMClassifier( boosting_type = 'gbdt',
                objective = 'binary',
                metric = 'auc',
                learning_rate = 0.01,
                #'is_unbalance': 'true', # replaced with scale_pos_weight argument
                num_leaves = 31,  # 2^max_depth - 1
                max_depth = -1,  # -1 means no limit
                min_child_samples = 20,  # Minimum number of data need in a child(min_data_in_leaf)
                max_bin = 255,  # Number of bucketed bin for feature values
                subsample = 0.6,  # Subsample ratio of the training instance.
                subsample_freq = 0,  # frequence of subsample, <=0 means no enable
                colsample_bytree = 0.3,  # Subsample ratio of columns when constructing each tree.
                min_child_weight = 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
                subsample_for_bin = 200000,
                min_split_gain = 0,
                reg_alpha = 0,
                reg_lambda = 0,
                scale_pos_weight =300 # because training data is extremely unbalanced 
                )
}

## LE + XGBoost
est_XGB_class = [xgb.XGBClassifier(  objective='binary:logistic'
                                    , eta=0.3
                                    , tree_method='hist'
                                    , grow_policy='lossguide'
                                    , max_leaves = 1400
                                    , max_depth = 0
                                    , subsample = 0.9
                                    , colsample_bytree = 0.7
                                    , colsample_bylevel = 0.7
                                    , min_child_weight = 0
                                    , alpha = 4
                                    , scale_pos_weight = 9
                                    , eval_metric = 'auc'
                                    , random_state = 9
                                    , silent = True
                                    #, n_jobs = 4
                                    #, learning_rate=0.01
                                    , n_estimators=10
                                    #, seed = 1234
                                    )
#                , xgb.XGBClassifier( objective='binary',
#                                    , learning_rate=0.01
#                                    , n_estimators=100
#                                    , seed = 1234
#                                    )
                ]
xgb_est_base = xgb.XGBClassifier( objective='binary:logistic'
                                 , tree_method='hist'
                                 , grow_policy='lossguide'
                                 , eval_metric='auc'
                                 , random_state = 9
                                )
