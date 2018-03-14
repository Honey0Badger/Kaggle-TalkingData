from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from matplotlib import pyplot
import seaborn as sns

# path to where the data lies
dpath = './data/'
train = pd.read_csv(dpath +"Otto_train.csv")
#train.head()

# drop ids and get labels
y_train = train['target']
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s)-1)

train = train.drop(["id", "target"], axis=1)
X_train = np.array(train)

# prepare cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)


def modelfit(alg, X_train, y_train, cv_folds=None, early_stopping_rounds=10):
    xgb_param = alg.get_xgb_params()
    xgb_param['num_class'] = 9


    xgtrain = xgb.DMatrix(X_train, label=y_train)

    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=cv_folds,
                      metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)

    cvresult.to_csv('1_nestimators.csv', index_label='n_estimators')


    n_estimators = cvresult.shape[0]


    alg.set_params(n_estimators=n_estimators)
    alg.fit(X_train, y_train, eval_metric='mlogloss')

    # Predict training set:
    # train_predprob = alg.predict_proba(X_train)
    # logloss = log_loss(y_train, train_predprob)

# Print model report:
# print ("logloss of train :" )
# print logloss

xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        objective= 'multi:softprob',
        seed=3)

modelfit(xgb1, X_train, y_train, cv_folds = kfold)



