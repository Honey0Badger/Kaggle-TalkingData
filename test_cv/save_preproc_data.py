import os
import sys
import psutil
import datetime
import time
import random
import numpy as np
import math

import pandas as pd


now = datetime.datetime.now()
print("Print timestamp for record...")
print(now.strftime("%Y-%m-%d %H:%M"))
sys.stdout.flush()

start = time.time()

train_test_df = pd.read_pickle('../stacking/pre_proc_inputs/f30_full_data.pkl')
train_len = 184903890
train_df = train_test_df[:train_len]

train_df['is_attributed'] = train_df['is_attributed'].astype('int8')
pos = train_df[train_df.is_attributed == 1]
neg = train_df[train_df.is_attributed == 0]

print("# of possible samples: ", len(pos))
print("# of negative samples: ", len(neg))

Nsample = 400000
p_set = pos.sample(Nsample)
n_set = neg.sample(Nsample)

full_df = pd.concat([p_set, n_set])
full_df = full_df.sample(frac=1).reset_index(drop=True) # shuffle the rows

print("*************************  Full data info **********************************\n")
full_df.info()
print("*************************  End of data info **********************************\n")
sys.stdout.flush()
            
process = psutil.Process(os.getpid())
print("- - - - - - - Memory usage check: ", process.memory_info().rss/1048576)
sys.stdout.flush()

full_df.to_csv("./f30_cv_sample.csv", index=False)
print("finished processing")
