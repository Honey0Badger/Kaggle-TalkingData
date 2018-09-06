from models import *

import numpy as np 
import pandas as pd 

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
