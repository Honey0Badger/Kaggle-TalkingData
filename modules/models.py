import xgboost as xgb
from pylightgbm.models import GBMRegressor


### LightGBM
est_GBM_reg = [GBMRegressor(exec_path="/users/cchen1/library/LightGBM/lightgbm",
                     learning_rate=0.01, ## use smaller learning rate for better accuracies
                     num_iterations=100000,
                     max_bin=<>,
                     num_leaves=<>,
                     min_data_in_leaf=<>,
                     feature_fraction=<>,
                     bagging_fraction=<>,
                     bagging_freq=1,
                     verbose = True),
              GBMRegressor(exec_path="/users/cchen1/library/LightGBM/lightgbm",
                     learning_rate=0.01, ## use smaller learning rate for better accuracies
                     num_iterations=100000,
                     max_bin=<>,
                     num_leaves=<>,
                     min_data_in_leaf=<>,
                     feature_fraction=<>,
                     bagging_fraction=<>,
                     bagging_freq=1,
                     verbose = True),
              GBMRegressor(exec_path="/users/cchen1/library/LightGBM/lightgbm",
                     learning_rate=0.01, ## use smaller learning rate for better accuracies
                     num_iterations=100000,
                     max_bin=<>,
                     num_leaves=<>,
                     min_data_in_leaf=<>,
                     feature_fraction=<>,
                     bagging_fraction=<>,
                     bagging_freq=1,
                     verbose = True),
              GBMRegressor(exec_path="/users/cchen1/library/LightGBM/lightgbm",
                     learning_rate=0.01, ## use smaller learning rate for better accuracies
                     num_iterations=100000,
                     max_bin=<>,
                     num_leaves=<>,
                     min_data_in_leaf=<>,
                     feature_fraction=<>,
                     bagging_fraction=<>,
                     bagging_freq=1,
                     verbose = True)
        ]

## LE + XGBoost
est_XGB_reg = [xgb.XGBRegressor(objective=logregobj,
                               learning_rate=0.01,
                               n_estimators=10000,
                               max_depth= <>,
                               min_child_weight = <>,
                               colsample_bytree = <>,
                               subsample = <>,
                               gamma = 1.0,
                               nthread = -1,
                               silent = True,
                               seed = 1234),
              xgb.XGBRegressor(objective=logregobj,
                               learning_rate=0.01,
                               n_estimators=10000,
                               max_depth= <>,
                               min_child_weight = <>,
                               colsample_bytree = <>,
                               subsample = <>,
                               gamma = 1.0,
                               nthread = -1,
                               silent = True,
                               seed = 1234),
              xgb.XGBRegressor(objective=logregobj,
                               learning_rate=0.01,
                               n_estimators=10000,
                               max_depth= <>,
                               min_child_weight = <>,
                               colsample_bytree = <>,
                               subsample = <>,
                               gamma = 1.0,
                               nthread = -1,
                               silent = True,
                               seed = 1234),
              xgb.XGBRegressor(objective=logregobj,
                               learning_rate=0.01,
                               n_estimators=10000,
                               max_depth= <>,
                               min_child_weight = <>,
                               colsample_bytree = <>,
                               subsample = <>,
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
