

import xgboost as xgb

from scipy import sparse
from sklearn.model_selection import train_test_split
from scipy.stats import skew, boxcox
from sklearn import preprocessing

import gc
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math


def load_from_file(filepath):
        data = pd.read_csv(filepath)
        print ("Loading data finished...")
        return data

def sample_inputs(infile, outfile):
        data = pd.read_csv(infile)
        sample = data.head(5000)
        sample.to_csv(outfile)
        

def train_test_data(full_data, cat_cols, num_cols, train_size):
        lift = 200
        full_cols = num_cols + cat_cols
        train_x = full_data[full_cols][:train_size].values
        test_x = full_data[full_cols][train_size:].values
        train_y = np.log(full_data[:train_size].loss.values + lift)
        ID = full_data.id[:train_size].values
        return train_x, train_y, test_x

def sparse_train_test_data(sparse_data, full_data,  num_cols, train_size):
        lift = 200
        full_data_sparse = sparse.hstack((sparse_data
                                         ,full_data[num_cols])
                                         ,format='csr'
                                        )
        train_x = sparse_data[:train_size]
        test_x = sparse_data[train_size:]
        train_y = np.log(full_data[:train_size].loss.values + lift)
        return train_x, train_y, test_x

def process_trainData(train):
        train_ts = pd.to_datetime(train['click_time'])
        train_y = train.is_attributed
        train.drop(['click_time', 'attributed_time', 'is_attributed']
                    , axis=1, inplace=True)
        train = train.assign(weekday=train_ts.dt.weekday_name.values)
        train = train.assign(hour=train_ts.dt.hour.values)
        cat_cols = ['weekday', 'hour']
        for cat_col in cat_cols:
                print ("Factorize feature %s" % (cat_col))
                train[cat_col] = preprocessing.LabelEncoder().fit_transform(train[cat_col])

        return train.values, train_y.values
        
def process_testData(test):
        test_ts = pd.to_datetime(test['click_time'])
        test.drop(['click_time']
                    , axis=1, inplace=True)
        test = test.assign(weekday=test_ts.dt.weekday_name.values)
        test = test.assign(hour=test_ts.dt.hour.values)

        cat_cols = ['weekday', 'hour']
        for cat_col in cat_cols:
                print ("Factorize feature %s" % (cat_col))
                test[cat_col] = preprocessing.LabelEncoder().fit_transform(test[cat_col])

        return test.values

def data_features(data):
        data_types = data.dtypes
        cat_cols = list(data_types[data_types=='object'].index)
        num_cols = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)
        id_col = 'id'
        target_col = 'loss'
        num_cols.remove('id')
        num_cols.remove('loss')

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
        skewed_cols = data[num_cols].apply(lambda x: skew(x.dropna()))
        print(skewed_cols.sort_values())

        #** Apply box-cox transformations: **
        skewed_cols = skewed_cols[skewed_cols > 0.25].index.values
        for col in skewed_cols:
                data[col], lam = boxcox(data[col] + 1)

        #** Apply Standard Scaling:**
        for col in num_cols:
                data[[col]] = preprocessing.StandardScaler().fit_transform(data[[col]])
        return data

def train_valid_split(train_x, train_y):
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, train_size=0.8, random_state=1234)
        xgtrain = xgb.DMatrix(train_x, label=train_y, missing=np.nan)
        return X_train, X_val, y_train, y_val, xgtrain

        



