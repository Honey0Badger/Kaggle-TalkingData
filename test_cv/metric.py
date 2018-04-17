from sklearn import metrics
from sklearn.metrics import mean_absolute_error

import numpy as np


def log_mae(labels, preds, lift=200):
    return mean_absolute_error(np.exp(labels) - lift, np.exp(preds) - lift)

log_mae_scorer = metrics.make_scorer(log_mae, greater_is_better=False)

def logregobj(labels, preds):
    con = 2
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con ** 2 / (np.abs(x) + con) ** 2
    return grad, hess

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

