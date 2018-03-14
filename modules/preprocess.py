from sklearn import preprocessing

from scipy import sparse
from scipy.stats import skew, boxcox

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math


def loadData_from_file(train_file, test_file):
        train_data = pd.read_csv(train_file)
        print ("Loading train data finished...")
        test_data = pd.read_csv(test_file)
        print ("Loading test data finished...")
        return train_data, test_data


def merge_train_test(train_data, test_data):
        full_data=pd.concat([train_data, test_data])
        print ("Full Data set created.")
        return full_data


def data_features(data):
        data_types = data.dtypes
        cat_cols = list(data_types[data_types=='object'].index)
        num_cols = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)
        id_col = 'id'
        target_col = 'is_attributed'
        num_cols.remove('id')
        num_cols.remove('is_attributed')

        print ( "Categorical features:", cat_cols)
        print ( "Numerica features:", num_cols)
        print ( "ID: %s, target: %s" %( id_col, target_col))
        return id_col, target_col, cat_cols, num_cols
        

def category_encoding(data, cat_cols):
        for cat_col in cat_cols:
                print ("Factorize feature %s" % (cat_col))
                data[cat_col] = preprocessing.LabelEncoder().fit_transform(data[cat_col])
        print ('Label enconding finished...')
        return data
        

def one_hot_encoding(data, cat_cols):
        OHE = preprocessing.OneHotEncoder(sparse=True)
        data_sparse = OHE.fit_transform(data[cat_cols])
        print ('One-hot-encoding finished...')
        print ('data sparse shape:', data_sparse.shape)
        return data_sparse

def process_num_data(data, num_cols):
    """ Numeric features
        We will apply two preprocessings on numeric features:
        1. Apply box-cox transformations for skewed numeric features.
        2. Scale numeric features so they will fall in the range between 0 and 1.
        """
        skewed_cols = data[num_cols].apply(lambda x: skew(x.dropna()))
        print (skewed_cols.sort_values())

        #** Apply box-cox transformations: **
        skewed_cols = skewed_cols[skewed_cols > 0.25].index.values
        for col in skewed_cols:
                data[col], lam = boxcox(data[col] + 1)

        #** Apply Standard Scaling:**
        for col in num_cols:
                data[col] = preprocessing.StandardScaler().fit_transform(data[col])
        return data

def num_cat_train_valid_split(data, cat_cols, num_cols, train_size):
        lift = 200
        full_cols = num_cols + cat_cols
        train_x = data[full_cols][:train_size].values
        test_x = data[full_cols][train_size:].values
        train_y = np.log(data[:train_size].loss.values + lift)
        ID = data.id[:train_size].values
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=0.8, random_state=1234)
        xgtrain = xgb.DMatrix(train_x, label=train_y, missing=np.nan)
        return X_train, X_val, y_train, y_val, xgtrain


def num_OHE_train_valid_split(data_sparse, cat_cols, num_cols, train_size):
        lift = 200
        full_data_sparse = sparse.hstack((data_sparse
                                         ,data[num_cols])
                                         ,format='csr'
                                        )
        print (full_data_sparse.shape)
        train_x = full_data_sparse[:train_size]
        test_x = full_data_sparse[train_size:]
        train_y = np.log(full_data[:train_size].loss.values + lift)
        ID = full_data.id[:train_size].values
        xgtrain = xgb.DMatrix(train_x, label=train_y,missing=np.nan) #used for Bayersian Optimization
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=.80, random_state=1234)
        return X_train, X_val, y_train, y_val, xgtrain
        

def batch_generator(X, y, batch_size, shuffle):
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




# Choose your best 4 models. Feel free to add more as long as their performance are close enough to the best one.


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





### MLP blend function

early_stop = EarlyStopping(monitor='val_mae_log', patience=5, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath="../tmp/weights.hdf5", monitor='val_mae_log', verbose=0, save_best_only=True,
                               mode='min')





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

