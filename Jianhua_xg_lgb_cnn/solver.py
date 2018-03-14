import xgboost as xgb
import pandas as pd
from sklearn import preprocessing, pipeline, metrics, model_selection
# grid_search, cross_validation, GridSearchCV,
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import grid_search
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from pylightgbm.models import GBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold

from scipy import sparse
from scipy.stats import skew, boxcox

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD,Nadam
#from keras.regularizers import WeightRegularizer, ActivityRegularizer,l2, activity_l2

##comment out the following two lines if you are using theano
import tensorflow as tf
tf.python.control_flow_ops = tf


def log_mae(labels, preds, lift=200):
    return mean_absolute_error(np.exp(labels) - lift, np.exp(preds) - lift)


# mean_absolute_error
def logregobj(labels, preds):
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con ** 2 / (np.abs(x) + con) ** 2
    return grad, hess


log_mae_scorer = metrics.make_scorer(log_mae, greater_is_better=False)


def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
    ##Grid Search for the best model
    model = model_selection.GridSearchCV(estimator=est,
                                         param_grid=param_grid,
                                         scoring=log_mae_scorer,
                                         verbose=10,
                                         n_jobs=n_jobs,
                                         iid=True,
                                         refit=refit,
                                         cv=cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("Scores:", model.grid_scores_)
    return model


def xg_eval_mae(yhat, dtrain, lift=200):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y) - lift, np.exp(yhat) - lift)


def xgb_logregobj(preds, dtrain):
    con = 2
    labels = dtrain.get_label()
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con ** 2 / (np.abs(x) + con) ** 2
    return grad, hess


def search_model_mae(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
    ##Grid Search for the best model
    model = model_selection.GridSearchCV(estimator=est,
                                         param_grid=param_grid,
                                         scoring='neg_mean_absolute_error',
                                         verbose=10,
                                         n_jobs=n_jobs,
                                         iid=True,
                                         refit=refit,
                                         cv=cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("Scores:", model.cv_results_)
    return model


## XGBoost blending function


def xgb_blend(estimators, train_x, train_y, test_x, fold, early_stopping_rounds=0):
    print("Blend %d estimators for %d folds" % (len(estimators), fold))
    skf = list(KFold(len(train_y), fold))

    train_blend_x = np.zeros((train_x.shape[0], len(estimators)))
    test_blend_x = np.zeros((test_x.shape[0], len(estimators)))
    scores = np.zeros((len(skf), len(estimators)))
    best_rounds = np.zeros((len(skf), len(estimators)))

    for j, est in enumerate(estimators):
        print("Model %d: %s" % (j + 1, est))
        test_blend_x_j = np.zeros((test_x.shape[0], len(skf)))
        for i, (train, val) in enumerate(skf):
            print("Model %d fold %d" % (j + 1, i + 1))
            fold_start = time.time()
            train_x_fold = train_x[train]
            train_y_fold = train_y[train]
            val_x_fold = train_x[val]
            val_y_fold = train_y[val]
            if early_stopping_rounds == 0:  # without early stopping
                est.fit(train_x_fold, train_y_fold)
                best_rounds[i, j] = est.n_estimators
                val_y_predict_fold = est.predict(val_x_fold)
                score = log_mae(val_y_fold, val_y_predict_fold, 200)
                print("Score: ", score)
                scores[i, j] = score
                train_blend_x[val, j] = val_y_predict_fold
                test_blend_x_j[:, i] = est.predict(test_x)
                print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))
            else:  # early stopping
                est.set_params(n_estimators=10000)
                est.fit(train_x_fold,
                        train_y_fold,
                        eval_set=[(val_x_fold, val_y_fold)],
                        eval_metric=xg_eval_mae,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                        )
                best_round = est.best_iteration
                best_rounds[i, j] = best_round
                print("best round %d" % (best_round))
                val_y_predict_fold = est.predict(val_x_fold, ntree_limit=best_round)
                score = log_mae(val_y_fold, val_y_predict_fold, 200)
                print("Score: ", score)
                scores[i, j] = score
                train_blend_x[val, j] = val_y_predict_fold
                test_blend_x_j[:, i] = est.predict(test_x, ntree_limit=best_round)
                print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))

        test_blend_x[:, j] = test_blend_x_j.mean(1)
        print("Score for model %d is %f" % (j + 1, np.mean(scores[:, j])))
    print("Score for blended models is %f" % (np.mean(scores)))
    return (train_blend_x, test_blend_x, scores, best_rounds)


## LightGBM blending function
def gbm_blend(estimators, train_x, train_y, test_x, fold, early_stopping_rounds=0):
    print("Blend %d estimators for %d folds" % (len(estimators), fold))
    skf = list(KFold(len(train_y), fold))

    train_blend_x = np.zeros((train_x.shape[0], len(estimators)))
    test_blend_x = np.zeros((test_x.shape[0], len(estimators)))
    scores = np.zeros((len(skf), len(estimators)))
    best_rounds = np.zeros((len(skf), len(estimators)))

    for j, gbm_est in enumerate(estimators):
        print("Model %d: %s" % (j + 1, gbm_est))
        test_blend_x_j = np.zeros((test_x.shape[0], len(skf)))
        params = gbm_est.get_params()
        for i, (train, val) in enumerate(skf):
            print("Model %d fold %d" % (j + 1, i + 1))
            est = GBMRegressor()
            est.param = params
            #             est.exec_path='/users/cchen1/library/LightGBM/lightgbm'
            est.exec_path = '/Users/Jianhua/anaconda/lightgbm'
            print(est)
            fold_start = time.time()
            train_x_fold = train_x[train]
            train_y_fold = train_y[train]
            val_x_fold = train_x[val]
            val_y_fold = train_y[val]
            if early_stopping_rounds == 0:  # without early stopping
                est.fit(train_x_fold, train_y_fold)
                best_rounds[i, j] = est.num_iterations
                val_y_predict_fold = est.predict(val_x_fold)
                score = log_mae(val_y_fold, val_y_predict_fold, 200)
                print("Score: ", score, mean_absolute_error(val_y_fold, val_y_predict_fold))
                scores[i, j] = score
                train_blend_x[val, j] = val_y_predict_fold
                test_blend_x_j[:, i] = est.predict(test_x)
                print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))
            else:  # early stopping
                est.set_params(num_iterations=1000000)
                est.set_params(early_stopping_round=early_stopping_rounds)
                est.set_params(verbose=False)
                est.fit(train_x_fold,
                        train_y_fold,
                        test_data=[(val_x_fold, val_y_fold)]
                        )
                best_round = est.best_round
                best_rounds[i, j] = best_round
                print("best round %d" % (best_round))
                val_y_predict_fold = est.predict(val_x_fold)
                score = log_mae(val_y_fold, val_y_predict_fold, 200)
                print("Score: ", score, mean_absolute_error(val_y_fold, val_y_predict_fold))
                scores[i, j] = score
                train_blend_x[val, j] = val_y_predict_fold
                test_blend_x_j[:, i] = est.predict(test_x)
                print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))

        test_blend_x[:, j] = test_blend_x_j.mean(1)
        print("Score for model %d is %f" % (j + 1, np.mean(scores[:, j])))
    print("Score for blended models is %f" % (np.mean(scores)))
    return (train_blend_x, test_blend_x, scores, best_rounds)


## Load Data
start = time.time()
train_data = pd.read_csv('../input/train.csv')
train_size=train_data.shape[0]
print ("Loading train data finished in %0.3fs" % (time.time() - start))

test_data = pd.read_csv('../input/test.csv')
print ("Loading test data finished in %0.3fs" % (time.time() - start))


## Merge train and test

#This will save our time on duplicating logics for train and test and will also ensure
#the transformations applied on train and test are the same.

full_data=pd.concat([train_data
                       ,test_data])
del( train_data, test_data)
print ("Full Data set created.")

## Group features

#In this step we will group the features into different groups so we can preprocess them seperately afterward.

data_types = full_data.dtypes
cat_cols = list(data_types[data_types=='object'].index)
num_cols = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)

id_col = 'id'
target_col = 'is_attributed'
num_cols.remove('id')
num_cols.remove('is_attributed')

print ("Categorical features:", cat_cols)
print ( "Numerica features:", num_cols)
print ( "ID: %s, target: %s" %( id_col, target_col))

## Categorical features
### 1. Label Encoding (Factorizing)

LBL = preprocessing.LabelEncoder()
start=time.time()
for cat_col in cat_cols:
    print ("Factorize feature %s" % (cat_col))
    full_data[cat_col] = LBL.fit_transform(full_data[cat_col])
print ('Label enconding finished in %f seconds' % (time.time()-start))

### 2. One Hot Encoding (get dummies)

#OHE can be done by either Pandas' get_dummies() or SK Learn's OneHotEncoder.

#1) get_dummies is easier to implement (can be used directly on raw categorical features,
# i.e. strings, but it takes longer time and is not memory efficient.

#2) OneHotEncoder requires the features being converted to numeric, which has already been done
# by LabelEncoder in previous step, and is much more efficient (7x faster).

#3) We will convert the OHE's results to a sparse matrix which uses way less memory as compared
# to dense matrix. However, not all algorithms and packagers support sparse matrix, e.g. Keras.
# In that case, we'll need to use other tricks to make it work.

OHE = preprocessing.OneHotEncoder(sparse=True)
start=time.time()
full_data_sparse=OHE.fit_transform(full_data[cat_cols])
print ('One-hot-encoding finished in %f seconds' % (time.time()-start))

print (full_data_sparse.shape)

## Numeric features

#We will apply two preprocessings on numeric features:

#1. Apply box-cox transformations for skewed numeric features.

#2. Scale numeric features so they will fall in the range between 0 and 1.

#Please be advised that these preprocessings are not necessary for tree-based models, e.g.
# XGBoost. However, linear or linear-based models, which will be dicussed in following weeks, may benefit from them.


#** Calculate skewness of each numeric features: **

skewed_cols = full_data[num_cols].apply(lambda x: skew(x.dropna()))
print (skewed_cols.sort_values())

#** Apply box-cox transformations: **
skewed_cols = skewed_cols[skewed_cols > 0.25].index.values
for skewed_col in skewed_cols:
    full_data[skewed_col], lam = boxcox(full_data[skewed_col] + 1)

#** Apply Standard Scaling:**
SSL = preprocessing.StandardScaler()
for num_col in num_cols:
    full_data[num_col] = SSL.fit_transform(full_data[num_col])


#### Note: LBL and OHE are likely exclusive so we will use one of them at a time combined
# with numeric features. In the following steps we will use OHE + Numeric to tune XGBoost
# models and you can apply the same process with OHE + Numeric features.
# Averaging results from two different models will likely generate better results.
####

## Numberic features + Label-encoded categorical features

# Initialize data
lift = 200

full_cols = num_cols + cat_cols
train_x = full_data[full_cols][:train_size].values
test_x = full_data[full_cols][train_size:].values
train_y = np.log(full_data[:train_size].loss.values + lift)
ID = full_data.id[:train_size].values

xgtrain = xgb.DMatrix(train_x, label=train_y,missing=np.nan) #used for Bayersian Optimization

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=.80, random_state=1234)

### LightGBM
## Choose your best 4 models. Feel free to add more as long as their performance are close enough to the best one.

estimators = [GBMRegressor(exec_path="/users/cchen1/library/LightGBM/lightgbm",
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

(train_blend_x_gbm_le,
 test_blend_x_gbm_le,
 blend_scores_gbm_le,
 best_rounds_gbm_le) = gbm_blend(estimators, train_x, train_y, test_x,
                                 4,
                                 500) #as the learning rate decreases the number of stopping rounds need to be increased

print (np.mean(blend_scores_gbm_le,axis=0))
print (np.mean(best_rounds_gbm_le,axis=0))
np.savetxt("../input/train_blend_x_gbm_le.csv",train_blend_x_gbm_le, delimiter=",")
np.savetxt("../input/test_blend_x_gbm_le.csv",test_blend_x_gbm_le, delimiter=",")

## LE + XGBoost
estimators = [xgb.XGBRegressor(objective=logregobj,
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

(train_blend_x_xgb_le,
 test_blend_x_xgb_le,
 blend_scores_xgb_le,
 best_rounds_xgb_le) = xgb_blend(estimators,
                                 train_x,
                                 train_y,
                                 test_x,
                                 4,
                                 500)

print(np.mean(blend_scores_xgb_le, axis=0))
print(np.mean(best_rounds_xgb_le, axis=0))
np.savetxt("../input/train_blend_x_xgb_le.csv", train_blend_x_xgb_le, delimiter=",")
np.savetxt("../input/test_blend_x_xgb_le.csv", test_blend_x_xgb_le, delimiter=",")

## Numberic features + One-hot-encoded categorical features
lift = 200

full_data_sparse = sparse.hstack((full_data_sparse
                                  ,full_data[num_cols])
                                 , format='csr'
                                 )
print (full_data_sparse.shape)
train_x = full_data_sparse[:train_size]
test_x = full_data_sparse[train_size:]
train_y = np.log(full_data[:train_size].loss.values + lift)
ID = full_data.id[:train_size].values

xgtrain = xgb.DMatrix(train_x, label=train_y,missing=np.nan) #used for Bayersian Optimization

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=.80, random_state=1234)

### OHE + LightGBM
estimators = [GBMRegressor(exec_path="/users/cchen1/library/LightGBM/lightgbm",
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


(train_blend_x_gbm_ohe,
 test_blend_x_gbm_ohe,
 blend_scores_gbm_ohe,
 best_rounds_gbm_ohe) = gbm_blend(estimators, train_x, train_y, test_x,
                                 4,
                                 500)

print (np.mean(blend_scores_gbm_ohe,axis=0))
print (np.mean(best_rounds_gbm_ohe,axis=0))
np.savetxt("../input/train_blend_x_gbm_ohe.csv",train_blend_x_gbm_ohe, delimiter=",")
np.savetxt("../input/test_blend_x_gbm_ohe.csv",test_blend_x_gbm_ohe, delimiter=",")

### OHE +XGBoost
estimators = [xgb.XGBRegressor(objective=logregobj,
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

(train_blend_x_xgb_ohe,
 test_blend_x_xgb_ohe,
 blend_scores_xgb_ohe,
 best_rounds_xgb_ohe) = xgb_blend(estimators,
                                  train_x,
                                  train_y,
                                  test_x,
                                  4,
                                  1000)

print(np.mean(blend_scores_xgb_ohe, axis=0))
print(np.mean(best_rounds_xgb_ohe, axis=0))
np.savetxt("../input/train_blend_x_xgb_ohe.csv", train_blend_x_xgb_ohe, delimiter=",")
np.savetxt("../input/test_blend_x_xgb_ohe.csv", test_blend_x_xgb_ohe, delimiter=",")

### OHE + MLP
def mae_log(y_true, y_pred):
    return K.mean(K.abs((K.exp(y_pred)-200) - (K.exp(y_true)-200)))



def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

### MLP blend function

early_stop = EarlyStopping(monitor='val_mae_log', patience=5, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath="../tmp/weights.hdf5", monitor='val_mae_log', verbose=0, save_best_only=True,
                               mode='min')


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


def nn_blend_data(parameters, train_x, train_y, test_x, fold, early_stopping_rounds=0, batch_size=128):
    print("Blend %d estimators for %d folds" % (len(parameters), fold))
    skf = list(KFold(len(train_y), fold))

    train_blend_x = np.zeros((train_x.shape[0], len(parameters)))
    test_blend_x = np.zeros((test_x.shape[0], len(parameters)))
    scores = np.zeros((len(skf), len(parameters)))
    best_rounds = np.zeros((len(skf), len(parameters)))

    for j, nn_params in enumerate(parameters):
        print("Model %d: %s" % (j + 1, nn_params))
        test_blend_x_j = np.zeros((test_x.shape[0], len(skf)))
        for i, (train, val) in enumerate(skf):
            print("Model %d fold %d" % (j + 1, i + 1))
            fold_start = time.time()
            train_x_fold = train_x[train]
            train_y_fold = train_y[train]
            val_x_fold = train_x[val]
            val_y_fold = train_y[val]

            # early stopping
            model = nn_model(nn_params)
            print(model)
            fit = model.fit_generator(generator=batch_generator(train_x_fold, train_y_fold, batch_size, True),
                                      nb_epoch=70,
                                      samples_per_epoch=train_x_fold.shape[0],
                                      validation_data=(val_x_fold.todense(), val_y_fold),
                                      verbose=0,
                                      callbacks=[
                                          #                                                 EarlyStopping(monitor='val_mae_log'
                                          #                                                               , patience=early_stopping_rounds, verbose=0, mode='auto'),
                                          ModelCheckpoint(filepath="../tmp/weights.hdf5"
                                                          , monitor='val_mae_log',
                                                          verbose=1, save_best_only=True, mode='min')
                                      ]
                                      )

            best_round = sorted([[id, mae] for [id, mae] in enumerate(fit.history['val_mae_log'])], key=lambda x: x[1],
                                reverse=False)[0][0]
            best_rounds[i, j] = best_round
            print("best round %d" % (best_round))

            model.load_weights("../tmp/weights.hdf5")
            # Compile model (required to make predictions)
            model.compile(loss='mae', metrics=[mae_log], optimizer=nn_params['optimizer'])

            # print (mean_absolute_error(np.exp(y_val)-200, pred_y))
            val_y_predict_fold = model.predict_generator(generator=batch_generatorp(val_x_fold, batch_size, True),
                                                         val_samples=val_x_fold.shape[0]
                                                         )

            score = log_mae(val_y_fold, val_y_predict_fold, 200)
            print("Score: ", score, mean_absolute_error(val_y_fold, val_y_predict_fold))
            scores[i, j] = score
            train_blend_x[val, j] = val_y_predict_fold.reshape(val_y_predict_fold.shape[0])

            model.load_weights("../tmp/weights.hdf5")
            # Compile model (required to make predictions)
            model.compile(loss='mae', metrics=[mae_log], optimizer=nn_params['optimizer'])
            test_blend_x_j[:, i] = model.predict_generator(generator=batch_generatorp(test_x, batch_size, True),
                                                           val_samples=test_x.shape[0]
                                                           ).reshape(test_x.shape[0])
            print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))

        test_blend_x[:, j] = test_blend_x_j.mean(1)
        print("Score for model %d is %f" % (j + 1, np.mean(scores[:, j])))
    print("Score for blended models is %f" % (np.mean(scores)))
    return (train_blend_x, test_blend_x, scores, best_rounds)

bagging_num = 10
nn_parameters = []

nn_parameter =  { 'input_size' :400 ,
     'input_dim' : train_x.shape[1],
     'input_drop_out' : 0.5 ,
     'hidden_size' : 200 ,
     'hidden_drop_out' :0.3,
     'learning_rate': 0.1,
     'optimizer': 'adadelta'
    }

for i in range(bagging_num):
    nn_parameters.append(nn_parameter)



(train_blend_x_ohe_mlp,
 test_blend_x_ohe_mlp,
 blend_scores_ohe_mlp,
 best_round_ohe_mlp) = nn_blend_data(nn_parameters,
                                     train_x,
                                     train_y,
                                     test_x,
                                     4,
                                     5)

print (np.mean(blend_scores_ohe_mlp,axis=0))
print (np.mean(best_round_ohe_mlp,axis=0))
print ( log_mae(np.mean(train_blend_x_ohe_mlp,axis=1).reshape(train_size,1),train_y))
np.savetxt("../input/train_blend_x_ohe_mlp.csv",train_blend_x_ohe_mlp, delimiter=",")
np.savetxt("../input/test_blend_x_ohe_mlp.csv",test_blend_x_ohe_mlp, delimiter=",")

## Blending
#1. Ridge Regression
#  * Ridge is focused on finding out weight of each feature which is exactly what we are interested in.
#2. XGB linear

#Specifically, we will simply average predictions from MLP models before using them for blending.

# ridge
from sklearn.linear_model import ElasticNet,Ridge,LinearRegression
print  ("Blending.")
param_grid = {
    'alpha':[0,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,15,20,25,30,35,40,45,50,55,60,70]
              }
model = search_model(np.hstack((train_blend_x_gbm_le,
                                train_blend_x_xgb_le,
                                train_blend_x_xgb_ohe,
                                train_blend_x_gbm_ohe,
                                np.mean(train_blend_x_ohe_mlp,axis=1).reshape(train_size,1)))
                                         , train_y
                                         , Ridge()
                                         , param_grid
                                         , n_jobs=1
                                         , cv=4
                                         , refit=True)

print ("best subsample:", model.best_params_)

train_blend_x_ohe_mlp

# XGBoost gblinear
params = {
    'eta': 0.1,
    'booster': 'gblinear',
    'lambda': 0,
    'alpha': 0, # you can try different values for alpha
    'lambda_bias' : 0,
    'silent': 0,
    'verbose_eval': True,
    'seed': 1234
}

xgb.cv(params,
       xgb.DMatrix(np.hstack((train_blend_x_gbm_le,
                                train_blend_x_xgb_le,
                                train_blend_x_xgb_ohe,
                                train_blend_x_gbm_ohe,
                                np.mean(train_blend_x_ohe_mlp,axis=1).reshape(train_size,1)))
                   , label=train_y,missing=np.nan),
       num_boost_round=100000, nfold=4
                       , feval=xg_eval_mae,
             seed=1234,
             callbacks=[xgb.callback.early_stop(500)])


## Submission

pred_y_ridge = np.exp(model.predict(np.hstack((test_blend_x_gbm_le,
                                test_blend_x_xgb_le,
                                test_blend_x_xgb_ohe,
                                test_blend_x_gbm_ohe,
                                np.mean(train_blend_x_ohe_mlp,axis=1).reshape(test_x.shape[0],1))))) - lift

results = pd.DataFrame()
results['id'] = full_data[train_size:].id
results['loss'] = pred_y_ridge
results.to_csv("../output/sub_ridge_blended.csv", index=False)
print ("Submission created.")

params = {
    'eta': 0.1,
    'booster': 'gblinear',
    'lambda': 0,
    'alpha': 0, # you can try different values for alpha
    'lambda_bias' : 0,
    'silent': 0,
    'verbose_eval': True,
    'seed': 1234
}


xgtrain_blend = xgb.DMatrix(np.hstack((train_blend_x_gbm_le,
                                train_blend_x_xgb_le,
                                train_blend_x_xgb_ohe,
                                train_blend_x_gbm_ohe,
                                np.mean(train_blend_x_ohe_mlp,axis=1).reshape(train_size,1))),
                        label=train_y,missing=np.nan)

xgb_model=xgb.train(params,
                    xgtrain_blend,
                    num_boost_round=<best round of xgb.cv from above>,
                    feval=xg_eval_mae)

pred_y_gblinear = np.exp(xgb_model.predict(
        xgb.DMatrix(
            np.hstack((test_blend_x_gbm_le,
                       test_blend_x_xgb_le,
                       test_blend_x_xgb_ohe,
                       test_blend_x_gbm_ohe,
                       np.mean(test_blend_x_ohe_mlp,axis=1).reshape(test_x.shape[0],1)))
        )
    )
               ) - lift

results = pd.DataFrame()
results['id'] = full_data[train_size:].id
results['loss'] = pred_y_gblinear
results.to_csv("../output/sub_xgb_gblinear.csv", index=False)
print ("Submission created.")

## Final submission
#  weights: [0.5,0.5]

pred_y = pred_y_ridge*0.5 + pred_y_gblinear*0.5

results = pd.DataFrame()
results['id'] = full_data[train_size:].id
results['loss'] = pred_y
results.to_csv("../output/sub_final.csv", index=False)
print ("Submission created.")

