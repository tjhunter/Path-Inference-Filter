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
""" Trajectory representation for learning, and associated utilities for I/O.
"""
import numpy as np

class LearningTrajectory:
  """ Trajectory in the good format for learning.
  
  Useful fields:
    - features: list of list of vectors (a vector is itself a list).
    - connections: list of list of pairs.
    - features_np: numpy array, each row is a feature
    - num_choices
    
  This class is read only.
  """

  def __init__(self, features, connections):
    self.features = features
    self.connections = connections
    self.L = len(features)
    self.num_choices = [len(l) for l in self.features]
    # Numpy copy for performance.
    self.features_np = [np.array(feats) for feats in self.features]
    # Make sure 
    # Dictionary for performance:
    def build_backward(conns):
      ''' Builds fast indexing of backward connections (using a dictionary).
      '''
      d = {}
      for (i, j) in conns:
        if j not in d:
          d[j] = [i]
        else:
          d[j].append(i)
      return d
    self.connections_backward = [None] \
      + [build_backward(conns) for conns in self.connections]
    def build_forward(conns):
      ''' Builds fast indexing of forward connections (using a dictionary).
      '''
      d = {}
      for (i, j) in conns:
        if i not in d:
          d[i] = [j]
        else:
          d[i].append(j)
      return d
    self.connections_forward = [build_forward(conns) for conns \
         in self.connections]
    self.check_invariants()
  
  def check_invariants(self):
    ''' Checks a few invariants on the trajectory,
    Raises an exception if the invariants are not verified.
    '''
    assert(len(self.features) == self.L)
    assert(len(self.connections) == self.L - 1)
    for l in range(self.L-1):
      for (u, v) in self.connections[l]:
        assert(u >= 0), l
        assert(v >= 0), l
        assert(u < self.num_choices[l]), (l, (u, v))
        assert(v < self.num_choices[l+1]), (l, (u, v))
    for l in range(1, self.L):
      for v in self.connections_backward[l]:
        assert self.connections_backward[l][v], \
          (l, v, self.connections_backward[l])
    for l in range(self.L-1):
      for u in self.connections_forward[l]:
        assert self.connections_forward[l][u]
  
  def truncated(self, start=None, end=None):
    ''' The truncated version of a list.
    '''
    if start is None:
      start = 0
    if end is None:
      end = self.L
    return LearningTrajectory(self.features[start:end], \
                              self.connections[start:end-1])
