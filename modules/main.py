import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math


import tensorflow as tf

from preprocess import *
from models import *
from preprocess import *
from pipelines import *

tf.python.control_flow_ops = tf


start = time.time()

train_file = '../input/train.csv'
test_file = '../input/test.csv'

train_data, test_data = loadData_from_file(train_file, test_file)

full_data = merge_train_test(train_data, test_data)
del( train_data, test_data)
print ("Full Data set created.")

id_col, target_col, cat_cols, num_cols = data_features(full_data)
print ("feature extracted.")

full_data = category_encoding(full_data, cat_cols)
data_sparse = one_hot_encoding(full_data, cat_cols)

full_data = process_num_data(full_data, num_cols)
X_train, X_val, y_train, y_val, xgtrain = num_cat_train_valid_split(data, cat_cols, num_cols, train_size)

# lightgbm blend
(train_blend_x_gbm_le,
 test_blend_x_gbm_le,
 blend_scores_gbm_le,
 best_rounds_gbm_le) = gbm_blend(est_GBM_reg, train_x, train_y, test_x, len(est_GBM_reg,
                                 500) #as the learning rate decreases the number of stopping rounds need to be increased

print (np.mean(blend_scores_gbm_le,axis=0))
print (np.mean(best_rounds_gbm_le,axis=0))
np.savetxt("../input/train_blend_x_gbm_le.csv",train_blend_x_gbm_le, delimiter=",")
np.savetxt("../input/test_blend_x_gbm_le.csv",test_blend_x_gbm_le, delimiter=",")

# XGB blend
(train_blend_x_xgb_le,
 test_blend_x_xgb_le,
 blend_scores_xgb_le,
 best_rounds_xgb_le) = xgb_blend(est_XGB_reg, train_x, train_y, test_x, len(est_XGB_reg, 500)

print(np.mean(blend_scores_xgb_le, axis=0))
print(np.mean(best_rounds_xgb_le, axis=0))
np.savetxt("../input/train_blend_x_xgb_le.csv", train_blend_x_xgb_le, delimiter=",")
np.savetxt("../input/test_blend_x_xgb_le.csv", test_blend_x_xgb_le, delimiter=",")


X_train, X_val, y_train, y_val, xgtrain = num_OHE_train_valid_split(data_sparse, cat_cols, num_cols, train_size)

# lightgbm blend for OHE+num
(train_blend_x_gbm_ohe,
 test_blend_x_gbm_ohe,
 blend_scores_gbm_ohe,
 best_rounds_gbm_ohe) = gbm_blend(est_GBM_reg, train_x, train_y, test_x, len(est_GBM_reg), 500)

print (np.mean(blend_scores_gbm_ohe,axis=0))
print (np.mean(best_rounds_gbm_ohe,axis=0))
np.savetxt("../input/train_blend_x_gbm_ohe.csv",train_blend_x_gbm_ohe, delimiter=",")
np.savetxt("../input/test_blend_x_gbm_ohe.csv",test_blend_x_gbm_ohe, delimiter=",")

# XGB blend for OHE+num
(train_blend_x_xgb_ohe,
 test_blend_x_xgb_ohe,
 blend_scores_xgb_ohe,
 best_rounds_xgb_ohe) = xgb_blend(est_XGB_reg, train_x, train_y, test_x, len(est_XGB_reg), 1000)

print(np.mean(blend_scores_xgb_ohe, axis=0))
print(np.mean(best_rounds_xgb_ohe, axis=0))
np.savetxt("../input/train_blend_x_xgb_ohe.csv", train_blend_x_xgb_ohe, delimiter=",")
np.savetxt("../input/test_blend_x_xgb_ohe.csv", test_blend_x_xgb_ohe, delimiter=",")


# NN model
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
 best_round_ohe_mlp) = nn_blend_data(nn_parameters, train_x, train_y, test_x, len(nn_parameters), 5)

print (np.mean(blend_scores_ohe_mlp,axis=0))
print (np.mean(best_round_ohe_mlp,axis=0))
print ( log_mae(np.mean(train_blend_x_ohe_mlp,axis=1).reshape(train_size,1),train_y))
np.savetxt("../input/train_blend_x_ohe_mlp.csv",train_blend_x_ohe_mlp, delimiter=",")
np.savetxt("../input/test_blend_x_ohe_mlp.csv",test_blend_x_ohe_mlp, delimiter=",")
