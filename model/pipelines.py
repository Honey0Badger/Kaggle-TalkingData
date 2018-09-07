from models import *

import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV

def LGBM_model_full_train(train_X, train_y, test_X, params, cat_features):
    """ LightGBM model train on entire training set (no early stop,
        niterations specified by hand. This is for final submission.
    """
    print("predictors used:", list(train_X))
   
    train = lgb.Dataset(train_X.values, label=train_y.values,
                        categorical_feature=cat_features)
    del train_X

    # defining model parameters in the following block
    nboost = 100

    print("start training...\n")
    print("model param: ", params)
    bst = lgb.train(    params
                        , train
                        , num_boost_round=nboost
                        , valid_sets=[train]
                        , valid_names=['train']
                        , verbose_eval=10
                   )

    pred = bst.predict(test_X)

    return pred


def gridCV(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
    ##Grid Search for the best model
    model = GridSearchCV(estimator=est,
                         param_grid=param_grid,
                         scoring='roc_auc',
                         verbose=10,
                         n_jobs=n_jobs,
                         iid=True,
                         refit=refit,
                         cv=cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("params:\n")
    print(model.cv_results_.__getitem__('params'))
    print("mean test scores:\n")
    print(model.cv_results_.__getitem__('mean_test_score'))
    print("std test scores:\n")
    print(model.cv_results_.__getitem__('std_test_score'))
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("**********************************************")

    return model
