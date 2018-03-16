import xgboost as xgb

from scipy import sparse
from scipy.stats import skew, boxcox
from sklearn import preprocessing

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
        



