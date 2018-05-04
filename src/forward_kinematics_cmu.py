from __future__ import division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz_cmu
import time
import copy
import data_utils
import tensorflow as tf
import os

#define which action to visualize
# tf.app.flags.DEFINE_string("action", "eating", "specify action to visualize")
tf.app.flags.DEFINE_string("file", "my_model_cmu_samples_outlength25_load6000_nogan.h5", "which file to load")
tf.app.flags.DEFINE_string("sub_index", "_0", "which seed to visualize")
FLAGS = tf.app.flags.FLAGS


def fkl( angles, parent, offset, posInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 117

  # Structure that indicates parents for each joint
  njoints   = 38
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    # try:
    #     if not rotInd[i] : # If the list is empty
    #       xangle, yangle, zangle = 0, 0, 0
    #     else:
    #       xangle = angles[ rotInd[i][2]-1 ]
    #       yangle = angles[ rotInd[i][1]-1 ]
    #       zangle = angles[ rotInd[i][0]-1 ]
    # except:
    #    print (i)

    try:
        if not posInd[i] : # If the list is empty
          xangle, yangle, zangle = 0, 0, 0
        else:
          xangle = angles[ posInd[i][2]-1 ]
          yangle = angles[ posInd[i][1]-1 ]
          zangle = angles[ posInd[i][0]-1 ]
    except:
       print (i)

    r = angles[ expmapInd[i] ]

    thisRotation = data_utils.expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  xyz = xyz[:,[0,2,1]]

  return np.reshape( xyz, [-1] )

def revert_coordinate_space(channels, R0, T0):
  """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  # Loop through the passed posses
  for ii in range(n):
    R_diff = data_utils.expmap2rotmat( channels[ii, rootRotInd] )
    R = R_diff.dot( R_prev )

    channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
    T = T_prev + ((R_prev.T).dot( np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
    channels_rec[ii,:3] = T
    # T_prev = T
    # R_prev = R

  return channels_rec


def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([0, 1, 2, 3, 4, 5, 6,1, 8, 9,10, 11,12, 1, 14,15,16,17,18,19, 16,
                    21,22,23,24,25,26,24,28,16,30,31,32,33,34,35,33,37])-1

  offset = 70*np.array([0,0	,0	,0,	0,	0,	1.65674000000000,	-1.80282000000000,	0.624770000000000,	2.59720000000000,	-7.13576000000000,	0,	2.49236000000000,	-6.84770000000000,	0,	0.197040000000000,	-0.541360000000000,	2.14581000000000,	0,	0,	1.11249000000000,	0,	0,	0,	-1.61070000000000,	-1.80282000000000,	0.624760000000000,	-2.59502000000000,	-7.12977000000000,	0,	-2.46780000000000,	-6.78024000000000,	0,	-0.230240000000000,	-0.632580000000000,	2.13368000000000,	0,	0,	1.11569000000000,	0,	0,	0,	0.0196100000000000,	2.05450000000000,	-0.141120000000000,	0.0102100000000000,	2.06436000000000,	-0.0592100000000000,	0,	0,0,	0.00713000000000000,	1.56711000000000,	0.149680000000000,	0.0342900000000000,	1.56041000000000,	-0.100060000000000,	0.0130500000000000,	1.62560000000000,	-0.0526500000000000,	0,	0,	0,	3.54205000000000,	0.904360000000000,	-0.173640000000000,	4.86513000000000,	0,	0,	3.35554000000000,	0,	0	,0	,0	,0	,0.661170000000000,	0,	0,	0.533060000000000,	0,	0	,0	,0	,0	,0.541200000000000	,0	,0.541200000000000,	0	,0	,0	,-3.49802000000000,	0.759940000000000,	-0.326160000000000,	-5.02649000000000	,0	,0,	-3.36431000000000,	0,0,	0,	0	,0	,-0.730410000000000,	0,	0	,-0.588870000000000,0	,0,	0,	0	,0	,-0.597860000000000	,0	,0.597860000000000])
  offset = offset.reshape(-1,3)

  rotInd = [[6, 5, 4],
            [9, 8, 7],
            [12, 11, 10],
            [15, 14, 13],
            [18, 17, 16],
            [21, 20, 19],
            [],
            [24, 23, 22],
            [27, 26, 25],
            [30, 29, 28],
            [33, 32, 31],
            [36, 35, 34],
            [],
            [39, 38, 37],
            [42, 41, 40],
            [45, 44, 43],
            [48, 47, 46],
            [51, 50, 49],
            [54, 53, 52],
            [],
            [57, 56, 55],
            [60, 59, 58],
            [63, 62, 61],
            [66, 65, 64],
            [69, 68, 67],
            [72, 71, 70],
            [],
            [75, 74, 73],
            [],
            [78, 77, 76],
            [81, 80, 79],
            [84, 83, 82],
            [87, 86, 85],
            [90, 89, 88],
            [93, 92, 91],
            [],
            [96, 95, 94],
            []]
  posInd=[]
  for ii in np.arange(38):
      if ii==0:
          posInd.append([1,2,3])
      else:
          posInd.append([])


  expmapInd = np.split(np.arange(4,118)-1,38)

  return parent, offset, posInd, expmapInd

def main():
  actions =["walking","running","directing_traffic","soccer","basketball","washwindow","jumping","basketball_signal"]

  # Load all the data
  parent, offset, rotInd, expmapInd = _some_variables()
  for action in actions:
      # numpy implementation
      with h5py.File( '../Samples/samples_cmu/'+FLAGS.file, 'r' ) as h5f:
          # expmap_gt = h5f['expmap/gt/' + action + FLAGS.sub_index][40:50, :]
          # expmap_pred = h5f['expmap/preds/' + action + FLAGS.sub_index][:]
          expmap_gt = h5f['expmap/gt/' + action+FLAGS.sub_index][1:50, :]
          expmap_pred = h5f['expmap/preds/' + action+FLAGS.sub_index][:]

      nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]

      # Put them together and revert the coordinate space
      expmap_all = revert_coordinate_space( np.vstack((expmap_gt, expmap_pred)), np.array([[0,0,-1],[0,1,0],[0,0,-1]]), np.zeros(3) )
      expmap_gt   = expmap_all[:nframes_gt,:]
      expmap_pred = expmap_all[nframes_gt:,:]

      # Compute 3d points for each frame
      xyz_gt, xyz_pred = np.zeros((nframes_gt, 114)), np.zeros((nframes_pred, 114))
      for i in range( nframes_gt ):
        xyz_gt[i,:] = fkl( expmap_gt[i,:], parent, offset, rotInd, expmapInd )
      for i in range( nframes_pred ):
        xyz_pred[i,:] = fkl( expmap_pred[i,:], parent, offset, rotInd, expmapInd )

      # === Plot and animate ===
      fig = plt.figure(figsize=(6,12))
      ax = plt.gca(projection='3d')
      ob = viz_cmu.Ax3DPose(ax)

      path="visualize_ours/"+FLAGS.file+'/'+action
      # path = "visualize_ours/20171026/" + FLAGS.file + FLAGS.sub_index + '/' + action
      if not os.path.exists(path):
          os.makedirs(path)
      # Plot the conditioning ground truth
      for i in range(nframes_gt):

        ob.update( xyz_gt[i,:] )
        plt.show(block=False)
        fig.canvas.draw()
        # plt.axis('off')
        # plt.savefig(path+"/frame{}.png".format(i+10),bbox_inches='tight',transparent=True)
        plt.pause(0.01)

      # Plot the prediction
      for i in range(nframes_pred):

        ob.update( xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
        plt.show(block=False)
        fig.canvas.draw()
        # plt.axis('off')
        # plt.savefig(path+"/frame{}.png".format(i+20),bbox_inches='tight',transparent=True)
        plt.pause(0.01)


if __name__ == '__main__':
  main()
