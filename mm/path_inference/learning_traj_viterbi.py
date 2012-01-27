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
'''
Created on Oct 14, 2011

@author: tjhunter

Viterbi filter for trajectories: will find the most likely trace 
in a trajectory.
'''

import numpy as np
from numpy import dot, exp
from mm.path_inference.HardFilter import HardFilter

class TrajectoryViterbiRef(HardFilter):
  # pylint: disable=W0201
  """ Finds the most likely complete trajectory.
  
  Interesting fields:
    - traj: a Trajectory object
    - theta: the parameter vector
    - most_likely: list of indexes (as long as traj), each of the index
        is the element of the most likely trajectory at this step
    - most_likely_tree: list of indexes, for each element, gives the index of
        the previous element ending this point.
    - instant_most_likely: the start index of the most likely trajectory, up 
        to that index.
  Internal:
    - partial_probs: (list of lists) the probabilities of trajectories ending
        at each element of the trajectory. 
  """
  
  def __init__(self, traj, theta):
    HardFilter.__init__(self)
    self.traj = traj
    self.theta = theta
    if not self.traj.L:
      raise Exception("Empty trajectory")
    if self.theta.shape[0] != self.traj.features_np[0].shape[1]:
      raise Exception("Shape of theta is %s while shape of features is %s" % \
                      (self.theta.shape, self.traj.features_np[0].shape[1]))
    
  
  def computeProbabilities(self):
    """ COmpute probabilities (nothing to do for Viterbi).
    """
    pass
  
  def computeAssignments(self):
    self.computeMostLikely()
    self.assignments = self.most_likely
  
  # pylint: disable=W0201
  def computeMostLikely(self):
    """ Computes most_likely, most_likely_tree, partial_probs
    """
    if 'most_likely' in self.__dict__:
      return
    # Compute the weights
    self.ws = []
    for l in range(self.traj.L):
      self.ws.append(exp(dot(self.traj.features_np[l], self.theta)))
    # Initialization:
    self.partial_probs = []
    self.most_likely_tree = []
    self.partial_probs.append(self.ws[0])
    # Fill with some dummy values, so that the indexes match.
    self.most_likely_tree.append([None for _ in 
                                  range(self.traj.num_choices[0])])
    # Recursion along the trajectory:
    for l in range(1, self.traj.L):
      this_partial_probs = []
      this_most_likely_tree = []
      for i in range(self.traj.num_choices[l]):
        partial_prob = 0
        previous = None
        # Any backward connection to propagate to that point?
        if i in self.traj.connections_backward[l]:
          for j in self.traj.connections_backward[l][i]:
            new_prob = self.partial_probs[l-1][j] * self.ws[l][i]
            if new_prob > partial_prob:
              partial_prob = new_prob
              previous = j
        this_partial_probs.append(partial_prob)
        this_most_likely_tree.append(previous)
      self.partial_probs.append(this_partial_probs)
      self.most_likely_tree.append(this_most_likely_tree)
    self.instant_most_likely = [np.argmax(probs) for probs \
                                in self.partial_probs]
    # At the end, find the most likely partial traj
    idx = np.argmax(self.partial_probs[-1])
    # Trace back the most likely traj from the tree.
    most_likely = [idx]
    for l in range(self.traj.L-1, 0, -1):
      most_likely.append(self.most_likely_tree[l][most_likely[-1]])
    most_likely.reverse() # In place
    self.most_likely = most_likely


class TrajectoryViterbi1(TrajectoryViterbiRef):
  """ Finds the most likely complete trajectory, in a way that is safe against
  double underflows.
  
  Interesting fields:
    - traj: a Trajectory object
    - theta: the parameter vector
    - most_likely: list of indexes (as long as traj), each of the index
        is the element of the most likely trajectory at this step
    - most_likely_tree: list of indexes, for each element, gives the index of
        the previous element ending this point.
    - instant_most_likely: the start index of the most likely trajectory, up 
        to that index.
  Internal:
    - log_partial_probs: (list of lists) the log probabilities of trajectories ending
        at each element of the trajectory. (Unscaled values)
    - log_ws
  """

  # pylint: disable=W0201
  def computeMostLikely(self):
    """ Computes most_likely, most_likely_tree, log_partial_probs
    """
    if 'most_likely' in self.__dict__:
      return
    # Compute the weights
    self.log_ws = []
    for l in range(self.traj.L):
      self.log_ws.append(dot(self.traj.features_np[l], self.theta))
    # Initialization:
    self.log_partial_probs = []
    self.most_likely_tree = []
    self.log_partial_probs.append(self.log_ws[0])
    # Fill with some dummy values, so that the indexes match.
    self.most_likely_tree.append([None for _ in 
                                  range(self.traj.num_choices[0])])
    # Recursion along the trajectory:
    minf = float('-inf')
    for l in range(1, self.traj.L):
      this_partial_probs = []
      this_most_likely_tree = []
      for i in range(self.traj.num_choices[l]):
        log_partial_prob = minf
        previous = None
        if i in self.traj.connections_backward[l]:
          for j in self.traj.connections_backward[l][i]:
            new_prob = self.log_partial_probs[l-1][j] + self.log_ws[l][i]
            if new_prob > log_partial_prob:
              log_partial_prob = new_prob
              previous = j
        this_partial_probs.append(log_partial_prob)
        this_most_likely_tree.append(previous)
      self.log_partial_probs.append(this_partial_probs)
      self.most_likely_tree.append(this_most_likely_tree)
    self.instant_most_likely = [np.argmax(log_probs) for log_probs \
                                in self.log_partial_probs]
    # At the end, find the most likely partial traj
    idx = np.argmax(self.log_partial_probs[-1])
    # Trace back the most likely traj from the tree.
    most_likely = [idx]
    for l in range(self.traj.L-1, 0, -1):
      most_likely.append(self.most_likely_tree[l][most_likely[-1]])
    most_likely.reverse() # In place
    self.most_likely = most_likely

class TrajectoryViterbiRealTime1(TrajectoryViterbi1):
  """ Uses the instantaneous most likely element of the trajectory.
  
  Requires no backtracking, but may be wrong and may provide a physically
  disconnected trajectory.
  """

  def computeAssignments(self):
    self.computeMostLikely()
    self.assignments = self.instant_most_likely
  