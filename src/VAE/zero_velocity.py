

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import VAE
import numpy as np
import math
import os
import random
import sys
import time
import h5py

dataset=VAE.Data_cmu(50,25,'../data/cmu_mocap/train','../data/cmu_mocap/test')
one_hot=False
srnn_gts_euler=dataset.get_srnn_gts(one_hot, to_euler=True)
for action in dataset.actions:
    mean_errors=np.zeros((8,25))
    for i in np.arange(8):
        zero_velocity= np.copy(srnn_gts_euler[action][i][49:50,:])
        eulerchannels_pred=np.repeat(zero_velocity,25,axis=0)
        srnn_gts_euler[action][i][:, 0:6] = 0
        eulerchannels_pred[:, 0:6] = 0
        idx_to_use = np.where(np.std(srnn_gts_euler[action][i], 0) > 1e-4)[0]


        euc_error = np.power(srnn_gts_euler[action][i][50:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt(euc_error)
        mean_errors[i, :] = euc_error

    mean_mean_errors = np.mean(mean_errors, 0)
    print()
    print(action)
    print()
    print("{0: <16} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
        print(" {0:5d} |".format(ms), end="")
    print()

    print("{0: <16} |".format(action), end="")
    for ms in [1, 3, 7, 9, 13, 24]:
        if 25 >= ms + 1:
            print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
        else:
            print("   n/a |", end="")
    print()

    with h5py.File('./zero_velociy_error.h5', 'a') as hf:

        node_name = 'mean_{0}_error'.format(action)
        hf.create_dataset(node_name, data=mean_mean_errors)