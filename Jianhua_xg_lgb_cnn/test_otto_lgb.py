import lightgbm as lgbm
#from lgbm.sklearn import LGBMClassifier

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from matplotlib import pyplot
import json
import seaborn as sns

# path to where the data lies
dpath = './data/'
train = pd.read_csv(dpath +"Otto_train.csv")


# drop ids and get labels
y_train = train['target']
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s)-1)

train = train.drop(["id", "target"], axis=1)
X_train = np.array(train)


def modelfit(params, alg, X_train, y_train, early_stopping_rounds=10):
    lgbm_params = params.copy()
    lgbm_params['num_class'] = 9

    lgbmtrain = lgbm.Dataset(X_train, y_train, silent=True)

    cv_result = lgbm.cv(
        lgbm_params, lgbmtrain, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='multi_logloss',
        early_stopping_rounds=early_stopping_rounds, show_stdv=True, seed=0)
    # note: cv_results will look like: {"multi_logloss-mean": <a list of historical mean>,
    # "multi_logloss-stdv": <a list of historical standard deviation>}
    print('best n_estimators:', len(cv_result['multi_logloss-mean']))
    print('best cv score:', cv_result['multi_logloss-mean'][-1])
    # cv_result.to_csv('lgbm1_nestimators.csv', index_label = 'n_estimators')
    json.dump(cv_result, open('lgbm_1.json', 'w'))

    alg.set_params(n_estimators=len(cv_result['multi_logloss-mean']))
    alg.fit(X_train, y_train)

params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'nthread': -1,
          'learning_rate': 0.1,
          'num_leaves': 80,
          'max_depth': 5,
          'max_bin': 127,
          'subsample_for_bin': 50000,
          'subsample': 0.8,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 1,
          'reg_lambda': 0,
          'min_split_gain': 0.0,
          'min_child_weight': 1,
          'min_child_samples': 20,
          'scale_pos_weight': 1}

#          'silent': True,

lgbm1 = lgbm.sklearn.LGBMClassifier(num_class= 9, n_estimators=1000, seed=0, **params)

modelfit(params,lgbm1, X_train, y_train)