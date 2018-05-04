
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import sklearn.preprocessing as data_preprocessing
import data_utils

#-----------------data process for human3.6m dataset--------------------
class DataLoader(object):
    """
    The Data Loader for human action recognition.

    :param seq_length_in: length of input sequence  
    :param seq_length_out: length of output sequence
    """
    def __init__(self,
                 seq_length_in,
                 seq_length_out,
                 data_dir):
        self.actions=["walking", "eating", "smoking", "discussion", "directions",
                       "greeting", "phoning", "posing", "purchases", "sitting",
                       "sittingdown", "takingphoto", "waiting", "walkingdog",
                       "walkingtogether"]

        self.label_cvt = data_preprocessing.LabelEncoder()
        self.label_cvt.fit(self.actions)
        self.seq_length_in=seq_length_in
        self.seq_length_out=seq_length_out

        self.read_all_data(self.actions, data_dir, False) 

    def define_actions(self,action):
        """
        Define the list of actions we are using.
        
        Args
        action: String with the passed action. Could be "all"
        Returns
        actions: List of strings of actions
        Raises
        ValueError if the action is not included in H3.6M
        """

        if action in self.actions:
            return [action]
        
        if action == "all":
            return self.actions
        
        if action == "all_srnn":
            return ["walking", "eating", "smoking", "discussion"]
        
        raise (ValueError, "Unrecognized action: %d" % action)
    def read_all_data( self, actions, data_dir, one_hot=False):
        """
        Loads data for training/testing and normalizes it.
        
        Args
        actions: list of strings (actions) to load
        seq_length_in: number of frames to use in the burn-in sequence
        seq_length_out: number of frames to use in the output sequence
        data_dir: directory to load the data from
        one_hot: whether to use one-hot encoding per action
        Returns
        train_set: dictionary with normalized training data
        test_set: dictionary with test data
        data_mean: d-long vector with the mean of the training data
        data_std: d-long vector with the standard dev of the training data
        dim_to_ignore: dimensions that are not used becaused stdev is too small
        dim_to_use: dimensions that we are actually using in the model
        """


        train_subject_ids = [1,6,7,8,9,11]
        test_subject_ids = [5]

        train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
        test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

        # Compute normalization stats
        data_mean, data_std,  dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

        # Normalize -- subtract mean, divide by stdev
        train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
        test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
        print("done reading data.")

        self.train_set = train_set
        self.test_set = test_set

        self.data_mean = data_mean
        self.data_std = data_std

        self.dim_to_ignore = dim_to_ignore
        self.dim_to_use = dim_to_use

        self.train_keys = list(self.train_set.keys())
        
        
    def get_train_batch(self, batch_size):
        """
        Get a batch from train set
        """

        chosen_keys = np.random.choice( len(self.train_keys), batch_size )

        # How many frames in total do we need?
        total_frames = self.seq_length_in + self.seq_length_out

        
        encoder_inputs  = []
        decoder_outputs = []
        yhat = np.zeros([batch_size, len(self.actions)])

        for i in xrange( batch_size ):
            the_key = self.train_keys[ chosen_keys[i] ]
            n, pts = self.train_set[ the_key ].shape
            idx = np.random.randint( 16, n-total_frames )

            # Select the data around the sampled points
            data_sel = self.train_set[ the_key ][idx:idx+total_frames ,:]
            currentAction = self.label_cvt.transform([the_key[1]])
            yhat[i, currentAction[0]]=1
            # Add the data
            encoder_inputs += [np.expand_dims(data_sel[0:self.seq_length_in-1, :], 0)]
            decoder_outputs += [np.expand_dims(data_sel, 0)]

        return np.expand_dims(np.concatenate(encoder_inputs, axis=0), 3), np.expand_dims(np.concatenate(decoder_outputs, axis=0), 3), yhat

    def find_indices_srnn( self, action ):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """
        
        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState( SEED )
        
        subject = 5
        subaction1 = 1
        subaction2 = 2
        
        T1 = self.test_set[ (subject, action, subaction1, 'even') ].shape[0]
        T2 = self.test_set[ (subject, action, subaction2, 'even') ].shape[0]
        prefix, suffix = 50, 100
        
        idx = []
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        return idx

    def get_test_batch(self, action):
        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                   "posing", "purchases", "sitting", "sittingdown", "smoking",
                   "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
        
        if not action in actions:
            raise ValueError("Unrecognized action {0}".format(action))

        frames = {}
        frames[ action ] = self.find_indices_srnn( action )

        batch_size = 8 # we always evaluate 8 seeds
        subject    = 5 # we always evaluate on subject 5
        source_seq_len = self.seq_length_in
        target_seq_len = self.seq_length_out

        total_frames = source_seq_len + target_seq_len
        
        seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

        encoder_inputs  = []
        decoder_outputs = []
        yhat = np.zeros([batch_size, len(self.actions)])
            

        for i in xrange( batch_size ):

            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = self.test_set[ (subject, action, subsequence, 'even') ]


            
            data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

            encoder_inputs += [np.expand_dims(data_sel[0:source_seq_len-1, :], axis=0)]
            decoder_outputs += [np.expand_dims(data_sel, axis=0)]
            currentAction = self.label_cvt.transform([action])

            yhat[i,currentAction[0]] = 1

        return np.expand_dims(np.concatenate(encoder_inputs, axis=0), 3), np.expand_dims(np.concatenate(decoder_outputs, axis=0),3), yhat

    def get_srnn_gts(self, one_hot, to_euler=True):
        """
        Get the ground truths for srnn's sequences, and convert to Euler angles.
        (the error is always computed in Euler angles).

        Args
          actions: a list of actions to get ground truths for.
          model: training model we are using (we only use the "get_batch" method).
          test_set: dictionary with normalized training data.
          data_mean: d-long vector with the mean of the training data.
          data_std: d-long vector with the standard deviation of the training data.
          dim_to_ignore: dimensions that we are not using to train/predict.
          one_hot: whether the data comes with one-hot encoding indicating action.
          to_euler: whether to convert the angles to Euler format or keep thm in exponential map

        Returns
          srnn_gts_euler: a dictionary where the keys are actions, and the values
            are the ground_truth, denormalized expected outputs of srnns's seeds.
        """
        srnn_gts_euler = {}

        for action in self.actions:

            srnn_gt_euler = []
            _,srnn_expmap, _  = self.get_test_batch(action)
            srnn_expmap = np.squeeze(srnn_expmap)
            # expmap -> rotmat -> euler
            for i in np.arange(srnn_expmap.shape[0]):
                denormed = data_utils.unNormalizeData(srnn_expmap[i, :, :], self.data_mean, self.data_std, self.dim_to_ignore, self.actions,
                                                      one_hot)

                if to_euler:
                    for j in np.arange(denormed.shape[0]):
                        for k in np.arange(0, 97, 3):
                            denormed[j, k:k + 3] = data_utils.rotmat2euler(
                                data_utils.expmap2rotmat(denormed[j, k:k + 3]))

                srnn_gt_euler.append(denormed);

            # Put back in the dictionary
            srnn_gts_euler[action] = srnn_gt_euler

        return srnn_gts_euler


    def compute_test_error(self,action, pred_pose,srnn_gts_expmap,srnn_gts_euler,one_hot,samples_fname):
        """
        Compute the test error 
        """

        pred_pose = np.squeeze(pred_pose)
        predict_expmap = data_utils.revert_output_format(pred_pose, self.data_mean, self.data_std,
                                                         self.dim_to_ignore,
                                                         self.actions, one_hot)
        
        mean_errors = np.zeros((len(predict_expmap), predict_expmap[0].shape[0]))
        
        for i in np.arange(8):
            
            eulerchannels_pred = np.copy(predict_expmap[i])
            for j in np.arange(eulerchannels_pred.shape[0]):
                for k in np.arange(0, 97, 3):
                    idx = [k, k+1, k+2]
                    eulerchannels_pred[j, idx] = data_utils.rotmat2euler(
                        data_utils.expmap2rotmat(predict_expmap[i][j, idx]))

            eulerchannels_pred[:, 0:6] = 0
            srnn_gts_euler[action][i][:, 0:6] = 0 ## Fixed by ZHEN, we also need
            # Pick only the dimensions with sufficient standard deviation. Others are ignored.
            # the below code is wrong!!!
            # After a look at the original code, I found that the code below must be fixed!!! (ZHEN)
            idx_to_use = np.where(np.std(srnn_gts_euler[action][i], 0) > 1e-4)[0]
            #print(idx_to_use)

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
        for ms in [1,3,7,9,13,24]:
            if self.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
            else:
              print("   n/a |", end="")
        print()

        with h5py.File(samples_fname, 'a') as hf:
            for i in np.arange(8):
                node_name = 'expmap/gt/{1}_{0}'.format(i, action)
                hf.create_dataset(node_name, data=srnn_gts_expmap[action][i])
                node_name = 'expmap/preds/{1}_{0}'.format(i, action)
                hf.create_dataset(node_name, data=predict_expmap[i])
            node_name = 'mean_{0}_error'.format(action)
            hf.create_dataset(node_name, data=mean_mean_errors)
