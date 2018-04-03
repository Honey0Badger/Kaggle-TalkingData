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

lgbm_est_base = lgb.LGBMClassifier(   learning_rate=0.01
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
                                       #, subsample_for_bin=200000
                                       , is_unbalance=True
                                       , verbose = 0 )


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
                                    #, learning_rate=0.01
                                    #, n_estimators=100
                                    #, seed = 1234
                                    )
#                , xgb.XGBClassifier( objective='binary',
#                                    , learning_rate=0.01
#                                    , n_estimators=100
#                                    , seed = 1234
#                                    )
                ]
