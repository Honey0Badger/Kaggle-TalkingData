import sys
import gc
import pandas as pd
import random
import numpy as np
import math


feature_name = 'new_feature'

train_df = load_half_file(ftrain)
len_train = len(train_df)
test_df = load_from_file(ftest)
full_df = train_df.append(test_df)

del train_df, test_df
gc.collect()

feature_df = do_app_click_freq( full_df, feature_name );

feature_df.to_csv("../features/"+feature_name+".csv"\
                  , index=False, float_format='%1.5f')

# helper functions and feature engineering functions
def load_from_file(filepath):
        data = pd.read_csv(filepath, parse_dates=['click_time'])
        print ("Loading full data finished...")
        return data

def load_half_file(filepath):
        data = pd.read_csv(filepath, skiprows=range(1,124903891)
                , nrows=60000000, parse_dates=['click_time'])
        print ("Loading part data finished...")
        return data

# feature lists:
def do_app_click_freq( full_df, feature_name ):
    gp = full_df[['ip', 'app']].groupby(by=['app'])[['ip']]\
            .agg(lambda x: float(len(x)) / len(x.unique())).reset_index()\
            .rename(index=str, columns={'ip': feature_name})
    return gp
