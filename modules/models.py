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
est_XGB_reg = [xgb.XGBRegressor(objective=logregobj,
                               learning_rate=0.01,
                               n_estimators=100,
                               gamma = 1.0,
                               nthread = -1,
                               silent = True,
                               seed = 1234),
              xgb.XGBRegressor(objective=logregobj,
                               learning_rate=0.01,
                               n_estimators=100,
                               gamma = 1.0,
                               nthread = -1,
                               silent = True,
                               seed = 1234),
              ]
