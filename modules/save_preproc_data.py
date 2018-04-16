import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd

from preprocess import *

now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

train_file = '../input/train.csv'
test_file = '../input/test.csv'
full_df, len_train, predictors = read_merge_process(train_file, ftest=test_file)
print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
sys.stdout.flush()
            
process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

full_df.to_pickle('18_feature_bot60m_data.pkl')
