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
''' Alpha beta smoother for a trajectory.

Created on Sep 2, 2011

@author: tjhunter
'''
import numpy as np
from numpy import dot, exp
from mm.log_math import lse
from mm.path_inference import SoftFilter

class TrajectorySmootherRef(SoftFilter):
  """ Runs the alpha/beta recursion on the trajectory.

  Reference implementation (unsafe against log underflows).
  
  Interesting fields computed:
  - probabilities
  """
  # pylint: disable=W0201
  def __init__(self, traj, theta):
    SoftFilter.__init__(self)
    self.traj = traj
    self.theta = theta
  
  def computeProbabilities(self):
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
    self.backward = [np.zeros(N_l, dtype=np.float64) \
                     for N_l in self.traj.num_choices]
    self.backward[-1].fill(1.0)
    for l in range(self.traj.L-2, -1, -1):
      N_l = self.traj.num_choices[l]
      qs = np.zeros(N_l)
      for i in range(N_l):
        if i in self.traj.connections_forward[l]:
          s = sum([self.backward[l+1][j] \
                   for j in self.traj.connections_forward[l][i]])
          qs[i] += self.log_ws[l][i] * s
      assert sum(qs) > 0, l
      self.backward[l] = qs / sum(qs)
    # Multplication:
    self.probabilities = []
    for l in range(self.traj.L):
      ps = self.forward[l] * self.backward[l]
      assert sum(ps) > 0
      self.probabilities.append(ps / sum(ps))

class TrajectorySmoother1(TrajectorySmootherRef):
  """ Runs the alpha/beta recursion on the trajectory.
  
  Secure implementation with respect to log underflows.
  
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
    self.log_backward[-1].fill(0.0)
    for l in range(self.traj.L-2, -1, -1):
      N_l = self.traj.num_choices[l]
      qs = -inf * np.ones(N_l)
      for i in range(N_l):
        if i in self.traj.connections_forward[l]:
          log_s = lse([self.log_backward[l+1][j] \
                       for j in self.traj.connections_forward[l][i]])
          qs[i] = self.log_ws[l][i] + log_s
      self.log_backward[l] = qs - lse(qs)
    # Product:
    self.probabilities = []
    self.log_probabilities = []
    for l in range(self.traj.L):
      log_ps = self.log_forward[l] + self.log_backward[l]
      log_ps -= max(log_ps)
      # Aggressively clamp the values that are too low to prevent an underflow
      log_ps[log_ps < -200] = -200
      ps = exp(log_ps)
      self.log_probabilities.append(log_ps - lse(log_ps))
      assert sum(ps) > 0, (l, log_ps, self.log_forward[l], self.log_backward[l])
      ps /= sum(ps)
      self.probabilities.append(ps)
  