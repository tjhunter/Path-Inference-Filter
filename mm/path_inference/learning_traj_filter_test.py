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
Test files for the filter.

Created on Sep 5, 2011

@author: tjhunter
'''
from mm.path_inference.learning_traj_smoother import TrajectorySmootherRef
from mm.path_inference.learning_traj_filter import TrajectoryFilterRef, \
  TrajectoryFilter1
from mm.path_inference.learning_traj_test import simple_traj1, \
 simple_traj2, simple_traj6
from mm.path_inference.learning_traj_smoother_test import check_probs
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')

def check_prob_fields(probs1, probs2):
  """ Compares the two probability sequences. """
  assert len(probs1) == len(probs2)
  for l in range(len(probs1)):
    assert np.abs(probs1[l] - probs2[l]).max() < 1e-3, \
      (l, probs1[l], probs2[l], probs1, probs2)

def test_filter_ref_1():
  """ test_filter_ref_1
  """
  traj = simple_traj1()
  theta = np.array([1.0, -1.0])
  filter_0 = TrajectoryFilterRef(traj, theta, 0)
  filter_0.computeProbs()
  # The forward probabilities should equal the probabilities
  check_prob_fields(filter_0.forward, filter_0.probabilities)
  # Run the filter in inneficient smooting mode
  filter_L = TrajectoryFilterRef(traj, theta, traj.L)
  filter_L.computeProbs()
  smoother = TrajectorySmootherRef(traj, theta)
  smoother.computeProbs()
  check_prob_fields(filter_L.forward, smoother.forward)
  check_prob_fields(filter_L.backward, smoother.backward)
  check_prob_fields(filter_L.probabilities, smoother.probabilities)

def test_filter_ref_2():
  """ test_filter_ref_2
  """
  traj = simple_traj2()
  theta = np.array([1.0, -1.0])
  filter_0 = TrajectoryFilterRef(traj, theta, 0)
  filter_0.computeProbs()
  # The forward probabilities should equal the probabilities
  check_prob_fields(filter_0.forward, filter_0.probabilities)
  # Run the filter in inneficient smooting mode
  filter_L = TrajectoryFilterRef(traj, theta, traj.L)
  filter_L.computeProbs()
  smoother = TrajectorySmootherRef(traj, theta)
  smoother.computeProbs()
  check_prob_fields(filter_L.forward, smoother.forward)
  check_prob_fields(filter_L.backward, smoother.backward)
  check_prob_fields(filter_L.probabilities, smoother.probabilities)

def test_filter_ref_traj6_1():
  """ test_filter_ref_traj6_1 """
  traj = simple_traj6()
  theta = np.array([-1.0, 1.0])
  for k in range(traj.L):
    filter_ref = TrajectoryFilterRef(traj, theta, k)
    filter_ref.computeProbs()
    filter_1 = TrajectoryFilter1(traj, theta, k)
    filter_1.computeProbs()
    check_probs(filter_1, filter_ref)

def test_filter_ref_traj1_1():
  """ test_filter_ref_traj1_1 """
  traj = simple_traj1()
  theta = np.array([-1.0, 1.0])
  for k in range(traj.L):
    filter_ref = TrajectoryFilterRef(traj, theta, k)
    filter_ref.computeProbs()
    filter_1 = TrajectoryFilter1(traj, theta, k)
    filter_1.computeProbs()
    check_probs(filter_1, filter_ref)

def test_filter_ref_traj2_1():
  """ test_filter_ref_traj2_1 """
  traj = simple_traj2()
  theta = np.array([-1.0, 1.0])
  for k in range(traj.L):
    filter_ref = TrajectoryFilterRef(traj, theta, k)
    filter_ref.computeProbs()
    filter_1 = TrajectoryFilter1(traj, theta, k)
    filter_1.computeProbs()
    check_probs(filter_1, filter_ref)

