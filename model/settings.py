from feature_engineer import *
from pipelines import *


#################   data preparation section   ###################
ID = 'click_id'
TARGET = 'is_attributed'

# this is NOT RAW data, this is processed data, put 
# in the same directory as engineered features
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

FEATURE_LIST = [
                 ( do_countuniq, [ ['ip'], 'channel', 'X0'                                  ]),
                 (  do_cumcount, [ ['ip', 'device', 'os'], 'app', 'CM1'                     ]),
                 ( do_countuniq, [ ['ip','day'], 'hour', 'CU2'                              ]),
                 ( do_countuniq, [ ['ip'], 'app', 'CU3'                                     ]),
                 ( do_countuniq, [ ['ip','app'], 'os', 'CU4'                                ]),
                 ( do_countuniq, [ ['ip'], 'device', 'CU5'                                  ]),
                 ( do_countuniq, [ ['app'], 'channel', 'CU6'                                ]),
                 (  do_cumcount, [ ['ip'], 'os', 'CU7'                                      ]),
                 (     do_count, [ ['ip','app'], 'ip_app_count'                             ]),
                 ( do_countuniq, [ ['ip','device','os'], 'app', 'CU8'                       ]),
                 (     do_count, [ ['ip','day','hour'], 'ip_tcount'                         ]),
                 (     do_count, [ ['ip','app'], 'ip_app_count'                             ]),
                 (     do_count, [ ['ip','app', 'os'], 'ip_app_os_count'                    ]),
                 (       do_var, [ ['ip','day','channel'], 'hour', 'ip_tchan_count'         ]),
                 (       do_var, [ ['ip','app','os'], 'hour', 'ip_app_os_var'               ]),
                 (       do_var, [ ['ip','app','channel'], 'day', 'ip_app_channel_var_day'  ]),
                 (      do_mean, [ ['ip','app','channel'], 'hour', 'ip_app_channel_mean_day']),
                 ( do_app_click_freq, []),
                 ( do_nextClick, [])
               ]

#################   end of data preparation section   ###################

#################   model and task section   ####################

## for model search with gridCV
GRIDCV_EST = lgb_est
GRIDCV_PARAM = param_grid
CV_FOLD = 4

## for train multiple model and make predictions
SELECT_FEATURES = ['X0', 'CU2', 'CU7']                           # set to None if all features are used
DROP_FEAT = ['click_time', 'epochtime', 'attributed_time']       # discard some features from original dataframe
CAT_FEATURES = None

MODEL_LIST = [
                 ( LGBM_model_full_train, [lgb_param1, CAT_FEATURES] ),
                 ( LGBM_model_full_train, [lgb_param2, CAT_FEATURES] )
             ]
#################   end of model and task section   ####################
