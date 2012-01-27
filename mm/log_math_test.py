'''
Copyright 2011, 2012 Timothy Hunter <tjhunter@eecs.berkeley.edu>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) version 3.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public 
License along with this library.  If not, see <http://www.gnu.org/licenses/>.
'''
# pylint: disable=W0105
'''
Created on Sep 2, 2011

@author: tjhunter
'''

from mm.log_math import lse, lse_vec, MINUS_INF
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')
from numpy import log

def log_safe(x):
  """ Ensure that log(0) == -inf and not an error. """
  if x == 0.0:
    return -float('inf')
  return log(x)

def test_1():
  """ test_1 """
  v = [1, 2, 0.0]
  lv = [log_safe(x) for x in v]
  assert abs(log(3) - lse(lv)) < 1e-6, (v, lv)

def test_lse_vec_1():
  ''' test_lse_vec_1 '''
  # Trying to add a zero vector with itself
  xs = [(MINUS_INF, np.zeros(2))]
  (u, v) = lse_vec(xs)
  assert u == MINUS_INF, u
  assert np.abs(v).max() == 0

def test_lse_vec_2():
  ''' test_lse_vec_2 '''
  # Trying to add a zero vector with itself
  xs = [(MINUS_INF, np.ones(2))]
  (u, v) = lse_vec(xs)
  assert u == MINUS_INF, u
  assert np.abs(v).max() == 0

def test_2():
  """ test_1 """
  inf = float("inf")
  xs = [(log(3.0), np.ones(2)), (2, np.zeros(2)), \
        (-inf, np.ones(2)), (-inf, np.zeros(2))]
  (norm, x) = lse_vec(xs)
  assert abs(norm - log(3)) < 1e-6, (norm, log(3), x)
  assert np.abs(x - 1).max() < 1e-6, (x)