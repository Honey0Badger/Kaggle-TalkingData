import xgboost as xgb
import lightgbm as lgb

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import PReLU
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K
#from keras.optimizers import SGD,Nadam

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
              xgb.XGBRegressor(objective=logregobj,
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
                               seed = 1234)
              ]
"""
# NN model
def nn_model(params):
    model = Sequential()
    model.add(Dense(params['input_size'], input_dim=params['input_dim']))

    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(params['input_drop_out']))

    model.add(Dense(params['hidden_size']))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(params['hidden_drop_out']))

    #     nadam = Nadam(lr=1e-4)
    nadam = Nadam(lr=params['learning_rate'])

    model.add(Dense(1))
    model.compile(loss='mae', metrics=[mae_log], optimizer=params['optimizer'])
    return (model)
"""
