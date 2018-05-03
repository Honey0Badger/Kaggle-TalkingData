"""
This script reconstruct the data from model*_step runs
"""

import numpy as np

# specify the overall parameters for the model
# check those parameters carefully before each run
debug = False
prefix = './tmp_data/lgbm_f21_cv'
#prefix = './tmp_data/lgbm_cv'
nfold = 4
train_len = 184903890
test_len = 18790469
#train_len = 49999
#test_len = 49999
outfile = './model_outputs/f21_lgbm_4fold_cv'

# iniatialize constructed array
train_oos_pred = np.zeros((train_len,))
test_pred = np.zeros((test_len, nfold))
score = np.zeros((nfold,))

# reconstruct train and test predictions
for fold in range(nfold):
    npzfile=prefix + '_' + str(fold) + '.npz'
    dat=np.load(npzfile)
    print('load from fold %d file...' % fold)
    if fold != dat['fold']:
        print('Error: file name and data ind not consistent!\n')
        exit()
    train_ind = dat['train_ind']
    train_oos_pred[train_ind] = dat['train_pred']
    test_pred[:, fold] = dat['test_pred']
    score[fold] = dat['score']

print('Scores for each fold: ', score)
print('Saving data to %s' % outfile)

if debug:
    np.savetxt(outfile+'_train_debug.csv', train_oos_pred, fmt='%.5f', delimiter=",")
    np.savetxt(outfile+'_test_debug.csv', np.mean(test_pred, axis=1), fmt='%.5f', delimiter=",")
else:
    np.savetxt(outfile+'_train.csv', train_oos_pred, fmt='%.5f', delimiter=",")
    np.savetxt(outfile+'_test.csv', np.mean(test_pred, axis=1), fmt='%.5f', delimiter=",")
