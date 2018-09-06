from feature_engineer import *
from pipelines import *


#################   data preparation section   ###################
ID = 'click_id'
target = 'is_attributed'

# this is NOT RAW data, this is processed data, put 
# in the same directory as engineered features
train_file = 'train.csv'
test_file = 'test.csv'

feature_list = [
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
select_features = ['X0', 'CU2', 'CU7']                           # set to None if all features are used
drop_feat = ['click_time', 'epochtime', 'attributed_time']       # discard some features from original dataframe
categorical = None

model_list = [
                 ( LGBM_model_full_train, [lgb_param1, categorical] ),
                 ( LGBM_model_full_train, [lgb_param2, categorical] )
             ]
#################   end of model and task section   ####################
