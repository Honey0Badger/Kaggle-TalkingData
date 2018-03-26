import xgboost as xgb
from pylightgbm.models import GBMClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD,Nadam


from metric import *

### LightGBM classifier
est_GBM_class = [GBMClassifier( learning_rate=0.1,
                                metric='auc',
                                num_leaves=7,
                                max_depth=3,
                                max_bin=100,
                                min_data_in_leaf=100,    # min_child_samples in lgb
                                bagging_freq=1,          # subsample_freq 
                                bagging_fraction=0.7,    # subsample
                                feature_fraction=0.7,
                                is_unbalance=True,
                                verbose = True ),
                 GBMClassifier( learning_rate=0.1,
                                metric = 'auc',
                                num_leaves=7,
                                max_depth=8,
                                max_bin=100,
                                min_data_in_leaf=100,
                                bagging_freq=1,
                                bagging_fraction=0.7,
                                is_unbalance=True,
                                verbose = True )
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
