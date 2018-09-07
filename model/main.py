from settings import *

import sys
import os
import gc
import time
import glob
import datetime
import click
import numpy as np 
import pandas as pd 

@click.group()
def cli():
    pass

@cli.command()
@click.argument('train_dat', type=click.Path(exists=True))
@click.argument('test_dat', type=click.Path(exists=True))
@click.argument('outdir')
def preprocess(train_dat, test_dat, outdir): 
        _preprocess(train_dat, test_dat, outdir) 

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
def feature_engineer(input_dir):
        _feature_engineer(input_dir)

@cli.command()
@click.argument('dat_dir')
@click.argument('outdir')
def model_train_predict(dat_dir, outdir):
        _model_train_predict(dat_dir, outdir)

@cli.command()
@click.argument('dat_dir')
def search_model(dat_dir):
        _search_model(dat_dir)


def _preprocess(train_dat, test_dat, outdir): 
        """ This function take raw data from train data, test data,
            and prepare data for feature engineering and modeling. 
            The processed data are stored in ourdir
        """

        train_df = pd.read_csv(train_dat, parse_dates=['click_time'])
        print ("Loading train data finished...")
        train_df.info()
        test_df = pd.read_csv(test_dat, parse_dates=['click_time'])
        print ("Loading test data finished...")
        test_df.info()

        # convert timestamp col to day and hour
        train_df['day'] = train_df.click_time.dt.day.astype('uint8')
        train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')
        train_df['epochtime'] = train_df.click_time.astype(np.int64) // 10 ** 9
        test_df['day'] = test_df.click_time.dt.day.astype('uint8')
        test_df['hour'] = test_df.click_time.dt.hour.astype('uint8')
        test_df['epochtime'] = test_df.click_time.astype(np.int64) // 10 ** 9

        
        train_df.to_csv(outdir + TRAIN_FILE, index=False)
        test_df.to_csv(outdir + TEST_FILE, index=False)


def _feature_engineer(input_dir):
        """ This function engineer extra features and save each
            feature in a separate file for fast test/assembly.
        """

        start_time = time.time()
        train_df = pd.read_csv(input_dir+TRAIN_FILE)
        ltrain = len(train_df)
        test_df = pd.read_csv(input_dir+TEST_FILE)
        full_df = train_df.append(test_df)
        print("combined data size: ", len(full_df))

        # create extra features from settings' feature list
        for feat, args in FEATURE_LIST:
            fstart = time.time()
            feat_df = feat(full_df, *args)
            print('feature'+feat.__name__+'finished in '
                    , (time.time()-fstart)/60, ' mins')
            colname = list(feat_df)
            if len(colname) != 1:
                print('warning: more than one feature returned')
            feat_df[:ltrain].to_csv(input_dir+'FEAT_'+colname[0]+'_train.csv', index=False)
            feat_df[ltrain:].to_csv(input_dir+'FEAT_'+colname[0]+'_test.csv', index=False)
        print('all features finished in ', (time.time()-start_time)/60, ' mins')


def _model_train_predict(data_dir, outdir):
        """ This function assemble all features and train on
            models specified in setttings. Each model makes
            individual prediction and output submission file.
        """
        train_df, test_df = assemble_data(data_dir)
        print("Data assemble finished.\n")
        print("train size: ", len(train_df))
        print("test size: ", len(test_df))

        print("\nStart modeling...\n")
        now = datetime.datetime.now()
        print("Print timestamp for record...")
        print(now.strftime("%Y-%m-%d %H:%M"))
        start_time = time.time()

        train_y = train_df[TARGET]
        DROP_FEAT.append([ID, TARGET])
        train_drop = [ c for c in DROP_FEAT if c in list(train_df)]
        test_drop = [ c for c in DROP_FEAT if c in list(test_df)]
        train_df.drop(train_drop, axis=1, inplace=True)
        test_sub = test_df[[ID]].copy(deep=True)
        test_df.drop(test_drop, axis=1, inplace=True)

        for model, args in MODEL_LIST:
                fstart = time.time()
                pred = model(train_df, train_y, test_df, *args)
                print('model' + model.__name__+'finished in '
                        , (time.time()-fstart)/60, ' mins')
                test_sub[TARGET] = pred
                test_sub.to_csv(outdir+model.__name__+'.csv', index=False, float_format='%1.5f')

        print("all model train and predict finished: %0.2f Minutes" 
                        %((time.time() - start_time)/60))

def _search_model(data_dir):
        """This function searches model parameters through gridCV"""
        train_df, _ = assemble_data(data_dir)
        print("Data assemble finished.\n")
        print("train size: ", len(train_df))

        train_y = train_df[TARGET]
        DROP_FEAT.append([ID, TARGET])
        train_drop = [ c for c in DROP_FEAT if c in list(train_df)]
        train_df.drop(train_drop, axis=1, inplace=True)

        gridCV(train_df, train_y, GRIDCV_EST, GRIDCV_PARAM, 1, CV_FOLD )


def assemble_data(data_dir):
        """This function assembles data from the data_dir
           and return train and test dataframe
        """
        train_df = pd.read_csv(data_dir+TRAIN_FILE)
        test_df = pd.read_csv(data_dir+TEST_FILE)
        ltrain = len(train_df)
        ltest = len(test_df)

        if SELECT_FEATURES:
            print('select a subset of features to load:', SELECT_FEATURES)
            feat_train = [ data_dir+'FEAT_'+f+'_train.csv' for f in SELECT_FEATURES ]
            feat_test = [ data_dir+'FEAT_'+f+'_test.csv' for f in SELECT_FEATURES ]
        else:
            print('load all features in the directory:', data_dir)
            feat_train = glob.glob(data_dir+"FEAT*train.csv")
            feat_test = glob.glob(data_dir+"FEAT*test.csv")

        print("debug: ", feat_train)

        for ftrain, ftest in zip(feat_train, feat_test):
            train_add = pd.read_csv(ftrain)
            test_add = pd.read_csv(ftest)
            if len(train_add) != ltrain or len(test_add) != ltest:
                print("Feature "+feat+" has different # of rows, ignore this feature")
                print("feature in train: ", len(train_add), " compare to ", ltrain)
                print("feature in test: ", len(test_add), " compare to ", ltest)
                continue
            else:
                train_df = pd.concat([train_df, train_add], axis=1)
                test_df = pd.concat([test_df, test_add], axis=1)

        return train_df, test_df



if __name__ == '__main__':
        cli()
