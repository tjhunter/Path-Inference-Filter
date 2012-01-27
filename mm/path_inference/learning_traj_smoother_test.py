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
Created on Sep 2, 2011

@author: tjhunter
'''
from mm.path_inference.learning_traj_smoother import (TrajectorySmootherRef,
                                                      TrajectorySmoother1)
from mm.path_inference.learning_traj_test import simple_traj1, \
 simple_traj2, simple_traj3, simple_traj6, simple_traj5
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')

def check_probs(smoother1, smoother2):
  """ Compares the two probability sequences. """
  assert len(smoother1.probabilities) == len(smoother2.probabilities)
  for l in range(len(smoother2.probabilities)):
    assert np.abs(smoother1.probabilities[l] \
                  - smoother2.probabilities[l]).max() < 1e-3, \
      (l, smoother1.probabilities[l], smoother2.probabilities[l])

def test_smoother_ref_traj1_1():
  """ test_smoother_ref_traj1_1 """
  traj = simple_traj1()
  theta = np.array([0.0, 0.0])
  smoother_ref = TrajectorySmootherRef(traj, theta)
  smoother_ref.computeProbs()
  smoother_1 = TrajectorySmoother1(traj, theta)
  smoother_1.computeProbs()
  check_probs(smoother_1, smoother_ref)

def test_smoother_ref_traj2_1():
  """ test_smoother_ref_traj2_1 """
  traj = simple_traj2()
  theta = np.array([1.0, -1.0])
  smoother_ref = TrajectorySmootherRef(traj, theta)
  smoother_ref.computeProbs()
  smoother_1 = TrajectorySmoother1(traj, theta)
  smoother_1.computeProbs()
  check_probs(smoother_1, smoother_ref)

def test_smoother_ref_traj5_1():
  """ test_smoother_ref_traj5_1 """
  traj = simple_traj5()
  theta = np.array([-1.0])
  smoother_ref = TrajectorySmootherRef(traj, theta)
  smoother_ref.computeProbs()
  smoother_1 = TrajectorySmoother1(traj, theta)
  smoother_1.computeProbs()
  check_probs(smoother_1, smoother_ref)

def test_smoother_ref_traj6_1():
  """ test_smoother_ref_traj6_1 """
  traj = simple_traj6()
  theta = np.array([-1.0, 1.0])
  smoother_ref = TrajectorySmootherRef(traj, theta)
  smoother_ref.computeProbs()
  smoother_1 = TrajectorySmoother1(traj, theta)
  smoother_1.computeProbs()
  check_probs(smoother_1, smoother_ref)

def test_smoother_ref_traj3_1():
  """ test_smoother_ref_traj3_1 .
  Just check if it can be computed and does not trigger underflow warnings. """
  traj = simple_traj3()
  theta = np.array([-1.0, 1.0])
  smoother_1 = TrajectorySmoother1(traj, theta)
  smoother_1.computeProbs()

def test_smoother_ref_traj3_2():
  """ test_smoother_ref_traj3_1 .
  Just check if it can be computed and does not trigger underflow warnings. """
  traj = simple_traj3()
  theta = np.array([-1.0, 1.0])
  
  smoother_1 = TrajectorySmootherRef(traj, theta)
  smoother_1.computeProbs()
  