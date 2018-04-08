import os
import datetime
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd

from preprocess import *
from models import *
from preprocess import *
from pipelines import *

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

train_file = '../input/train.csv'
test_file = '../input/test.csv'

## create a small sample for testing
#sample_inputs(train_file, '../input/train_small.csv')
#sample_inputs(test_file, '../input/test_small.csv')
#print("finish writing")
#exit()

train_data = load_from_file(train_file)
#train_data = load_half_file(train_file)
train_size = train_data.shape
train_data.info()

train_x, train_y = process_trainData(train_data)
print("train data size: ", train_size)
sys.stdout.flush()

del train_data
gc.collect()

test_data = load_from_file(test_file)
Id = test_data['click_id'].values
test_size=test_data.shape
test_x = process_testData(test_data)
print ("feature extracted.")
print("test data size: ", test_size)
test_data.info()
sys.stdout.flush()

del test_data
gc.collect()

## lightgbm blend
#(test_blend_y_gbm_le,
# blend_scores_gbm_le,
# best_rounds_gbm_le) = lgbm_blend(est_LGBM_test, train_x, train_y, test_x, 4, 20)

# xgboost blend
(test_blend_y_xgb_le,
 blend_scores_xgb_le,
 best_rounds_xgb_le) = xgb_blend(est_XGB_class, train_x, train_y, test_x, 2, 0)

del train_x, train_y, test_x
gc.collect()

print (np.mean(blend_scores_xgb_le,axis=0))
print (np.mean(best_rounds_xgb_le,axis=0))
sys.stdout.flush()
#np.savetxt("../output/test_blend_y_gbm_le.csv",test_blend_y_gbm_le, delimiter=",")


submission = pd.DataFrame()
submission['is_attributed'] = np.mean(test_blend_y_xgb_le, axis=1)
submission.insert(loc=0, column='click_id', value = Id)
submission.to_csv("../output/sub_xgb.csv", index=False, float_format='%1.5f')

