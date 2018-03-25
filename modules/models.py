import xgboost as xgb
from pylightgbm.models import GBMRegressor, GBMClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD,Nadam


from metric import *

### LightGBM classifier
est_GBM_class = [GBMClassifier( learning_rate=0.01,
                                metric = 'auc',
                                verbose = True ),
                 GBMClassifier( learning_rate=0.01,
                                metric = 'auc',
                                verbose = True )
                ]

### LightGBM
est_GBM_reg = [GBMRegressor( learning_rate=0.01, ## use smaller learning rate for better accuracies
                     num_iterations=100,
                     bagging_freq=1,
                     verbose = True),
              GBMRegressor( learning_rate=0.01, 
                     num_iterations=100,
                     bagging_freq=1,
                     verbose = True),
              GBMRegressor( learning_rate=0.01,
                     num_iterations=100,
                     bagging_freq=1,
                     verbose = True),
              GBMRegressor( learning_rate=0.01, 
                     num_iterations=100,
                     bagging_freq=1,
                     verbose = True)
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
