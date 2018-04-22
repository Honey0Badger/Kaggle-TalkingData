import xgboost as xgb

from scipy import sparse
from sklearn.model_selection import train_test_split
from scipy.stats import skew, boxcox
from sklearn import preprocessing

import sys
import gc
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math

train_df = pd.read_csv('../input/train.csv', parse_dates=['click_time'])

train_df['day'] = train_df.click_time.dt.day.astype('uint8')
train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')

day6_df = train_df[train_df['day'] == 6]
day7_df = train_df[train_df['day'] == 7]
day8_df = train_df[train_df['day'] == 8]
day9_df = train_df[train_df['day'] == 9]

day6_df.to_csv('../input/day6_train.csv', index=False)
day7_df.to_csv('../input/day7_train.csv', index=False)
day8_df.to_csv('../input/day8_train.csv', index=False)
day9_df.to_csv('../input/day9_train.csv', index=False)




