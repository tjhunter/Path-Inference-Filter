#Copyright 2011, 2012 Timothy Hunter <tjhunter@eecs.berkeley.edu>
#
#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Lesser General Public
#License as published by the Free Software Foundation; either
#version 2.1 of the License, or (at your option) version 3.
#
#This library is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#Lesser General Public License for more details.
#
#You should have received a copy of the GNU Lesser General Public 
#License along with this library.  If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=W0105
'''
Some operations to perform common procedures in log domain.

Created on Sep 2, 2011

@author: tjhunter
'''
from math import exp, log
import numpy as np

MINUS_INF = -float('inf')
INF = float('inf')
MINUS_LOG = -100

def lse(xs):
  """ Sum of scalars in log domain (result also returned in log domain).
  
  Input:
  - list of elements
  """
  if len(xs) == 0:
    return MINUS_INF
#  print xs
  x_max = max(xs)
  if x_max == MINUS_INF:
    return MINUS_INF
  return x_max + log(sum([exp(x-x_max) for x in xs]))

def lse_npy(xs):
  """ Sum of scalars in log domain (result also returned in log domain).
  
  Input:
  - 1-D numpy array
  """
  if xs.shape[0] == 0:
    return MINUS_INF
  x_max = np.max(xs)
  if x_max == MINUS_INF:
    return MINUS_INF
  return x_max + np.log(np.sum(np.exp(np.clip(xs-x_max, MINUS_LOG, INF))))


def lse_vect(xs):
  """ Works on a MxN array, will return a M array.
  """
  (M, N) = xs.shape
  if N == 0:
    return np.ones(M) * MINUS_INF
  xs_max = np.max(xs, axis=1)
  return xs_max + np.log(np.sum(np.exp(xs - \
                                       np.outer(xs_max, 
                                                np.ones(N))), axis=1))

def get_scaled_vector(v):
  """ Input: a numpy vector.
  Output a pair of (log(norm2), normalized vector).
  """
  norm = np.abs(v).max()
  if norm == 0:
    return (MINUS_INF, np.zeros_like(v))
  lnorm = log(norm)
  return (lnorm, v/norm)


def lse_vec(xs):
  """ Sum of vectors in log domain.
  A vector in log_domain is represented as a pair of (scaling factor, vector)
  
  """
  assert xs, xs
  norm_max = max([norm for (norm, v) in xs])
  x = np.zeros_like(xs[0][1])
  # Everything is zero, no need to continue.
  if norm_max == MINUS_INF:
    return (MINUS_INF, x)
  for (norm, v) in xs:
    if norm - norm_max > MINUS_LOG:
      x += exp(norm - norm_max) * v
  (n2, x_res) = get_scaled_vector(x)
  return (norm_max + n2, x_res)

def lse_vec_(xs):
  """ Sum of vectors in log domain.
  A vector in log_domain is represented as a pair of (scaling factor, vector)
  
  """
  norm_max = max([norm for (norm, v) in xs])
  x = np.zeros_like(xs[0][1])
  # Everything is zero, no need to continue.
  if norm_max == MINUS_INF:
    return (MINUS_INF, x)
  
  for (norm, v) in xs:
    if norm - norm_max > -50:
      x += exp(norm - norm_max) * v
  (n2, x_res) = get_scaled_vector(x)
  return (norm_max + n2, x_res)

def lse_vec_2(xs_dirs, xs_norms):
  """ Sum of vectors in log domain.
  A vector in log_domain is represented as a pair of (scaling factor, vector)
  
  Arguments:
  - xs_dirs: list of arrays
  - xs_norms: list
  """
  norm_max = max(xs_norms)
  x = np.zeros_like(xs_dirs[0])
  # Everything is zero, no need to continue.
  if norm_max == MINUS_INF:
    return (MINUS_INF, x)
  L = len(xs_dirs)
  i = 0
  while i < L:
    norm = xs_norms[i]
    if norm - norm_max > MINUS_LOG:
      v = xs_dirs[i]
      x += exp(norm - norm_max) * v
    i += 1
  max_norm_x = np.max(np.abs(x))
  return (norm_max + np.log(max_norm_x), x / max_norm_x)


def lse_vec_npy(xs_dirs, xs_norms):
  """ Sum of vectors in log domain.
  A vector in log_domain is represented as a pair of (scaling factor, vector)
  
  Arguments:
  - xs_dirs: N by M array
  - xs_norms: N array
  """
  norm_max = np.max(xs_norms)
  # Everything is zero, no need to continue.
  if norm_max == MINUS_INF:
    return (MINUS_INF, np.zeros_like(xs_dirs[0]))
  ws = np.exp(np.clip(xs_norms-norm_max, MINUS_LOG, INF))
  x = np.dot(xs_dirs.T, ws)
  max_norm_x = np.max(np.abs(x))
  return (norm_max + np.log(max_norm_x), x / max_norm_x)
  