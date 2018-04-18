from sklearn.metrics import roc_auc_score
from sklearn import  model_selection
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression

import lightgbm as lgb

#from keras.callbacks import EarlyStopping, ModelCheckpoint

import xgboost as xgb

from preprocess import *
from models import *
from metric import *

import sys
import gc
import numpy as np
import time

def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
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
    sys.stdout.flush()

    # saving all model scores
#    np.savez('../output/grid_model_scores.npz', model.cv_results_)

    return model


def Bayes_search(train_x, train_y, est, param_range, cv, iterations=100, n_jobs=1, refit=False):
    # Grid Search for the best model
    bayes_model = BayesSearchCV(estimator=est,
                          search_spaces=param_range,
                          scoring='roc_auc',
                          verbose=0,
                          n_jobs=n_jobs,
                          refit=refit,
                          n_iter = iterations,
                          random_state=42,
                          cv=cv)
    
    # define callback function
    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search (From kaggle NanoMathias kernal)"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_model.cv_results_)    

        # Get current parameters and the best parameters    
        best_params = pd.Series(bayes_model.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
                len(all_models),
                np.round(bayes_model.best_score_, 4),
                bayes_model.best_params_
        ))
                                        
        # Save all model results
        clf_name = bayes_model.estimator.__class__.__name__
        all_models.to_csv(clf_name+"_cv_results.csv")

    # Fit Grid Search Model
    bayes_model.fit(train_x, train_y, callback=status_print)

    return 0



def LGBM_DataSet(train_df, valid_df, predictors, target, cat_features):
    train = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                        feature_name=predictors,
                        categorical_feature=cat_features
                        )
    valid = lgb.Dataset(valid_df[predictors].values, label=valid_df[target].values,
                        feature_name=predictors,
                        categorical_feature=cat_features
                        )
    return train, valid


def single_LGBM_train(params, train, valid, metrics, early_stopping_rounds=20):
    evals_results = {}
    bst = lgb.train(params
                    , train
                    , valid_sets=[train, valid]
                    , valid_names=['train', 'valid']
                    , evals_result=evals_results 
                    , early_stopping_rounds=early_stopping_rounds
                    , verbose_eval=10
                    )
    n_estimators = bst.best_iteration
    print("\nModel Report:")
    print("N_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst

# xgboost single model module
def XGB_DMatrix(train_df, valid_df, features, target):
    Dtrain = xgb.DMatrix(train_df[features], label=train_df[target].values)
    Dvalid = xgb.DMatrix(valid_df[features], label=valid_df[target].values)
    return Dtrain, Dvalid

def XGB_Dtest(test_df, features):
    return xgb.DMatrix(test_df, feature_names=features)

def single_XGB_train(params, dtrain, dvalid, metrics, early_stopping_rounds=20):
    evals_results = {}
    bst = xgb.train(params
                    , dtrain
                    , evals=[(dtrain,'train'), (dvalid, 'valid')]
                    , evals_result=evals_results 
                    , early_stopping_rounds=early_stopping_rounds
                    , verbose_eval=True
                    )
    n_estimators = bst.best_iteration
    print("\nModel Report:")
    print("N_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst

## LightGBM blending
def lgbm_blend(estimators, train_x, train_y, test_x, fold, early_stopping_rounds=0):
    print("Blend %d estimators for %d folds" % (len(estimators), fold))
    sys.stdout.flush()
    skf  = list(KFold(fold).split(train_y))

    test_blend_y = np.zeros((test_x.shape[0], len(estimators)))
    scores = np.zeros((len(skf), len(estimators)))
    best_rounds = np.zeros((len(skf), len(estimators)))

    for j, est in enumerate(estimators):
        print("Model %d: %s" % (j + 1, est))
        test_blend_y_j = np.zeros((test_x.shape[0],))
        for i, (train, val) in enumerate(skf):
            print("Model %d fold %d" % (j + 1, i + 1))
            sys.stdout.flush()
            fold_start = time.time()
            if early_stopping_rounds == 0:  # without early stopping
                est.fit(train_x[train], train_y[train], eval_metric='auc')
                best_rounds[i, j] = -1
                val_y_predict_fold = est.predict_proba(train_x[val])[:,1]
                score = roc_auc_score(train_y[val], val_y_predict_fold)
                del val_y_predict_fold
                print("AUC score: ", score)
                scores[i, j] = score
                test_blend_y_j = test_blend_y_j + est.predict_proba(test_x)[:,1]
                print("Model %d fold %d fitting finished in %0.3fs" 
                        % (j + 1, i + 1, time.time() - fold_start))
                sys.stdout.flush()
            else:  # early stopping
                est.set_params(num_boost_round=1000)
                est.set_params(verbose=-1)
                est.fit(train_x[train],
                        train_y[train],
                        eval_set=[(train_x[val], train_y[val])],
                        eval_metric='auc',
                        early_stopping_rounds = early_stopping_rounds,
                        verbose=False
                       )
                best_rounds[i, j] = est.best_iteration_
                print("best round %d" % (best_rounds[i,j]))
                sys.stdout.flush()
                val_y_predict_fold = est.predict_proba(train_x[val], num_iteration = est.best_iteration_)[:,1]
                score = roc_auc_score(train_y[val], val_y_predict_fold)
                del val_y_predict_fold
                print("AUC score: ", score)
                scores[i, j] = score
                test_blend_y_j = test_blend_y_j + est.predict_proba(test_x, num_iteration = est.best_iteration_)[:,1]
                print("Model %d fold %d fitting finished in %0.3fs" 
                        % (j + 1, i + 1, time.time() - fold_start))
                sys.stdout.flush()
            gc.collect()

        test_blend_y[:, j] = test_blend_y_j / fold
        print("Score for model %d is %f" % (j + 1, np.mean(scores[:, j])))
        sys.stdout.flush()
    print("Score for blended models is %f" % (np.mean(scores)))
    sys.stdout.flush()
    return (test_blend_y, scores, best_rounds)


## XGBoost blending 

def xgb_blend(estimators, train_x, train_y, test_x, fold, early_stopping_rounds=0):
    print("Blend %d estimators for %d folds" % (len(estimators), fold))
    sys.stdout.flush()
    skf  = list(KFold(fold).split(train_y))

    test_blend_y = np.zeros((test_x.shape[0], len(estimators)))
    scores = np.zeros((len(skf), len(estimators)))
    best_rounds = np.zeros((len(skf), len(estimators)))

    for j, est in enumerate(estimators):
        print("Model %d: %s" % (j + 1, est))
        sys.stdout.flush()
        test_blend_y_j = np.zeros((test_x.shape[0],))
        for i, (train, val) in enumerate(skf):
            print("Model %d fold %d" % (j + 1, i + 1))
            sys.stdout.flush()
            fold_start = time.time()
            if early_stopping_rounds == 0:  # without early stopping
                est.fit(train_x[train], train_y[train], eval_metric='auc')
                best_rounds[i, j] = -1
                val_y_predict_fold = est.predict_proba(train_x[val])[:,1]
                score = roc_auc_score(train_y[val], val_y_predict_fold)
                del val_y_predict_fold
                print("AUC Score: ", score)
                scores[i, j] = score
                test_blend_y_j = test_blend_y_j + est.predict_proba(test_x)[:,1]
                print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))
                sys.stdout.flush()
            else:  # early stopping
                est.set_params(n_estimators=10000)
                est.fit(train_x[train],
                        train_y[train],
                        eval_set=[(train_x[val], train_y[val])],
                        eval_metric='auc',
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                        )
                best_round = est.best_iteration
                best_rounds[i, j] = best_round
                print("best round %d" % (best_round))
                sys.stdout.flush()
                val_y_predict_fold = est.predict_proba(train_x[val], ntree_limit=best_round)[:,1]
                score = roc_auc_score(train_y[val], val_y_predict_fold)
                print("AUC Score: ", score)
                del val_y_predict_fold
                scores[i, j] = score
                test_blend_y_j = test_blend_y_j + est.predict_proba(test_x, ntree_limit=best_round)[:,1]
                print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))
                sys.stdout.flush()
            gc.collect()
            
        test_blend_y[:, j] = test_blend_y_j / fold
        print("Score for model %d is %f" % (j + 1, np.mean(scores[:, j])))
        sys.stdout.flush()
    print("Score for blended models is %f" % (np.mean(scores)))
    sys.stdout.flush()
    return (test_blend_y, scores, best_rounds)



"""
def nn_blend_data(train_x, train_y, test_x, fold, early_stopping_rounds=0, batch_size=128):
    skf  = list(KFold(fold).split(train_y))
    # setup callbacks and checkpoint functions
    early_stop = EarlyStopping(monitor='val_mae_log', patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath="../tmp/weights.hdf5", monitor='val_mae_log', verbose=0, save_best_only=True,
                                   mode='min')
    bagging_num = 1
    parameters = []

    nn_parameter =  { 'input_size' :400, 'input_dim' : train_x.shape[1], 'input_drop_out':0.5,
                      'hidden_size' : 200, 'hidden_drop_out' :0.3, 'learning_rate': 0.1, 'optimizer': 'adadelta' }

    for i in range(bagging_num):
        parameters.append(nn_parameter)

    print("Blend %d estimators for %d folds" % (len(parameters), fold))

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
                                      nb_epoch=1,
                                      samples_per_epoch=train_x_fold.shape[0],
                                      validation_data=(val_x_fold.todense(), val_y_fold),
                                      verbose=0,
                                      callbacks=[
                                          # EarlyStopping(monitor='val_mae_log'
                                          #, patience=early_stopping_rounds, verbose=0, mode='auto'),
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
                                                         steps = val_x_fold.shape[0] / np.ceil(val_x_fold.shape[0]/batch_size)
                                                         )

            score = log_mae(val_y_fold, val_y_predict_fold, 200)
            print("Score: ", score, mean_absolute_error(val_y_fold, val_y_predict_fold))
            scores[i, j] = score
            train_blend_x[val, j] = val_y_predict_fold.reshape(val_y_predict_fold.shape[0])

            model.load_weights("../tmp/weights.hdf5")
            # Compile model (required to make predictions)
            model.compile(loss='mae', metrics=[mae_log], optimizer=nn_params['optimizer'])
            test_blend_x_j[:, i] = model.predict_generator(generator=batch_generatorp(test_x, batch_size, True),
                                                           steps = test_x.shape[0] / np.ceil(test_x.shape[0]/batch_size),
                                                           ).reshape(test_x.shape[0])
            print("Model %d fold %d fitting finished in %0.3fs" % (j + 1, i + 1, time.time() - fold_start))

        test_blend_x[:, j] = test_blend_x_j.mean(1)
        print("Score for model %d is %f" % (j + 1, np.mean(scores[:, j])))
    print("Score for blended models is %f" % (np.mean(scores)))
    return (train_blend_x, test_blend_x, scores, best_rounds)

"""
"""
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

def ridge_blend(train_models, test_models, train_y):
        lift = 200
        param_grid = { 'alpha':[0,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,
                                 0.03,0.1,0.3,1,3,10,15,20,25,30,35,40,45,50,55,60,70]}
        model = search_model(np.hstack(train_models)
                             , train_y
                             , Ridge()
                             , param_grid
                             , n_jobs=1
                             , cv=2
                             , refit=True)

        print ("best subsample:", model.best_params_)
       
        pred_y_ridge = np.exp(model.predict(np.hstack(test_models))) - lift
        return pred_y_ridge

def gblinear_blend(train_models, test_models, train_y):
        lift = 200
        params = { 'eta': 0.1, 'booster': 'gblinear', 'lambda': 0, 'alpha': 0,
                   'lambda_bias' : 0, 'silent': 0, 'verbose_eval': True, 'seed': 1234 }

        xgb.cv(params, xgb.DMatrix(np.hstack(train_models)
                                , label=train_y,missing=np.nan)
                                , num_boost_round=100, nfold=2
                                , feval=xg_eval_mae
                                , seed=1234
                                , callbacks=[xgb.callback.early_stop(50)])

        xgtrain_blend = xgb.DMatrix(np.hstack(train_models), label=train_y, missing=np.nan)

        xgb_model=xgb.train(params, xgtrain_blend,
                            num_boost_round=0,
                            feval=xg_eval_mae)

        pred_y_gblinear = np.exp(xgb_model.predict(xgb.DMatrix(np.hstack(test_models)))) - lift

        return pred_y_gblinear
"""
