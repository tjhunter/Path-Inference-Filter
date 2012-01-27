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
''' k-lagged filter for a trajectory.

Created on Sep 2, 2011

@author: tjhunter
'''
import numpy as np
from numpy import dot, exp
from mm.log_math import lse
from mm.path_inference import SoftFilter

class TrajectoryFilterRef(SoftFilter):
  """ Runs the k-lagged filtering on a trajectory..

  k == 0 corresponds to pure filtering.
  k == L is equivalent to running forward-backward smoothing
  (warning, in this case, performance is *terrible*).

  Reference implementation (unsafe against log over/underflows), only use
  for testing!
  
  Interesting fields computed:
  - probabilities
  """
  # pylint: disable=W0201, w0231
  def __init__(self, traj, theta, k=1):
    self.traj = traj
    self.theta = theta
    self.k = k
  
  def computeProbabilities(self):
    """ Compute probabilities.
    """
    self.computeProbs()
  
  def computeProbs(self):
    """ Compute probabilities.
    """
    if 'probabilities' in self.__dict__:
      return
    # Compute the weights
    self.log_ws = []
    for l in range(self.traj.L):
      self.log_ws.append(exp(dot(self.traj.features_np[l], self.theta)))
    # Forward pass
    # Nothing changes here.
    # Init:
    self.forward = [np.zeros(N_l) for N_l in self.traj.num_choices]
    self.forward[0] = self.log_ws[0] / sum(self.log_ws[0])
    for l in range(1, self.traj.L):
      N_l = self.traj.num_choices[l]
      qs = np.zeros(N_l)
      for i in range(N_l):
        if i in self.traj.connections_backward[l]:
          s = sum([self.forward[l-1][j] \
                   for j in self.traj.connections_backward[l][i]])
          qs[i] = self.log_ws[l][i] * s
      assert sum(qs) > 0
      self.forward[l] = qs / sum(qs)
    # Backward pass
    # A bit more complicated because we need to do it again for every chunk.
    self.backward = [np.zeros(N_l, dtype=np.float64) \
                     for N_l in self.traj.num_choices]
    for l in range(self.traj.L):
      self.backward[l] = self.computeBackward(l, min(l+self.k, self.traj.L-1))
    # Multplication:
    self.probabilities = []
    for l in range(self.traj.L):
      ps = self.forward[l] * self.backward[l]
      assert sum(ps) > 0
      self.probabilities.append(ps / sum(ps))

  def computeBackward(self, t, t_start):
    """ Computes backward recursion for index t, \
    starting at index t_start >= t.
    
    Does not modify the function. Returns the vector of backward values.
    """
    assert t_start >= t
    assert self.traj.L > t_start
    # Intermediate vectors
    vs = []
    # Initizaliation
    N_t_start = self.traj.num_choices[t_start]
    vs.append(np.ones(N_t_start, dtype=np.float64))
    for l in range(t, t_start)[::-1]:
      N_l = self.traj.num_choices[l]
      qs = np.zeros(N_l)
      for i in range(N_l):
        if i in self.traj.connections_forward[l]:
          s = sum([vs[-1][j] \
                   for j in self.traj.connections_forward[l][i]])
          qs[i] += self.log_ws[l][i] * s
      assert sum(qs) > 0, l
      vs.append(qs / sum(qs))
    return vs[-1]
    
    
class TrajectoryFilter1(TrajectoryFilterRef):
  """ Runs the k-lagged filtering on a trajectory..

  k == 0 corresponds to pure filtering.
  k == L is equivalent to running forward-backward smoothing
  (warning, in this case, performance is *terrible*).

  Reference implementation (unsafe against log over/underflows), only use
  for testing!
  
  Interesting fields computed:
  - probabilities
  """
  # pylint: disable=W0201

  def computeProbs(self):
    """ Compute probabilities.
    """
    if 'probabilities' in self.__dict__:
      return
    inf = float('inf')
    # Compute the weights
    self.log_ws = []
    for l in range(self.traj.L):
      self.log_ws.append(dot(self.traj.features_np[l], self.theta))
    # Forward pass
    # Init:
    self.log_forward = [np.zeros(N_l) for N_l in self.traj.num_choices]
    self.log_forward[0] = self.log_ws[0] - lse(self.log_ws[0])
    for l in range(1, self.traj.L):
      N_l = self.traj.num_choices[l]
      qs = -inf * np.ones(N_l)
      for i in range(N_l):
        if i in self.traj.connections_backward[l]:
          log_s = lse([self.log_forward[l-1][j] \
                       for j in self.traj.connections_backward[l][i]])
          qs[i] = self.log_ws[l][i] + log_s
      self.log_forward[l] = qs - lse(qs)
    # Backward pass
    self.log_backward = [np.zeros(N_l, dtype=np.float64) \
                     for N_l in self.traj.num_choices]
    # Backward pass
    # A bit more complicated because we need to do it again for every chunk.
    for l in range(self.traj.L):
      self.log_backward[l] = self.computeLogBackward(l, min(l+self.k, \
                                                            self.traj.L-1))
    # Product:
    self.probabilities = []
    self.log_probabilities = []
    for l in range(self.traj.L):
      log_ps = self.log_forward[l] + self.log_backward[l]
      log_ps -= max(log_ps)
      # Aggressively clamp the values that are too low to prevent an underflow
      log_ps[log_ps < -200] = -200
      self.log_probabilities.append(log_ps - lse(log_ps))
      ps = exp(log_ps)
      assert sum(ps) > 0, (l, log_ps, self.log_forward[l], self.log_backward[l])
      ps /= sum(ps)
      self.probabilities.append(ps)

  def computeLogBackward(self, t, t_start):
    """ Computes backward recursion for index t, \
    starting at index t_start >= t.
    
    Does not modify the function. Returns the vector of backward values.
    """
    assert t_start >= t
    assert self.traj.L > t_start
    inf = float('inf')
    # Intermediate vectors
    vs = []
    # Initizaliation
    N_t_start = self.traj.num_choices[t_start]
    vs.append(np.zeros(N_t_start, dtype=np.float64))
    for l in range(t, t_start)[::-1]:
      N_l = self.traj.num_choices[l]
      qs = -inf * np.ones(N_l)
      for i in range(N_l):
        if i in self.traj.connections_forward[l]:
          log_s = lse([vs[-1][j] \
                   for j in self.traj.connections_forward[l][i]])
          qs[i] = self.log_ws[l][i] + log_s
      vs.append(qs - lse(qs))
    return vs[-1]
    
  