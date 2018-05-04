from __future__ import division
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
from libc.math cimport sqrt, sin, cos
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

cdef float eps = np.finfo(np.float32).eps


def expmap2rotmat(np.ndarray r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  
  cdef float r0[3]
  r0[0] = r[0]
  r0[1] = r[1]
  r0[2] = r[2]
  cdef float theta = sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2])
  r0[0] /= (theta + eps)
  r0[1] /= (theta + eps)
  r0[2] /= (theta + eps)
  
  
  cdef np.ndarray r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  cdef np.ndarray r0xx = r0x - r0x.T
  cdef np.ndarray R = np.eye(3,3) + np.sin(theta)*r0xx + (1-np.cos(theta))*(r0xx).dot(r0xx);
  return R

def rotmat2quat(np.ndarray R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  cdef np.ndarray rotdiff = R - R.T;

  cdef float r0[3]
  r0[0] = -rotdiff[1,2]
  r0[1] =  rotdiff[0,2]
  r0[2] = -rotdiff[0,1]
  cdef float theta = sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2])
  cdef float sintheta = theta / 2;

  r0[0] /= (theta + eps)
  r0[1] /= (theta + eps)
  r0[2] /= (theta + eps)
  
  
  cdef float costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  cdef np.ndarray q      = np.zeros(4)
  cdef float sinhalftheta = sin(theta/2)
  
  q[0]   = cos(theta/2)
  q[1] = r0[0]*sinhalftheta
  q[2] = r0[1]*sinhalftheta
  q[3] = r0[2]*sinhalftheta
  
  return q


def cvt_expmap_to_quat(np.ndarray data_expmap):

    cdef np.ndarray res = np.zeros([data_expmap.shape[0], data_expmap.shape[1], 4], dtype=DTYPE)

    cdef int imax = data_expmap.shape[0]
    cdef int jmax = data_expmap.shape[1]

    for idx in range(imax):
        for jdx in range(jmax):
            res[idx, jdx, :] = rotmat2quat(expmap2rotmat(data_expmap[idx, jdx, :]))

    return res


def cvt_quat_to_expmap(np.ndarray data_quat):

    cdef np.ndarray res = np.zeros([data_quat.shape[0], data_quat.shape[1], 3], dtype=DTYPE)
    
