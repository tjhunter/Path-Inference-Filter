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

Test suite for viterbi.
'''

from mm.path_inference.learning_traj_test import (simple_traj1,
                                                  simple_traj2,
                                                  simple_traj6,
                                                  simple_traj7)
from mm.path_inference.learning_traj_viterbi import (TrajectoryViterbiRef,
                                                     TrajectoryViterbi1)
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')

def test_viterbi_ref_1():
  """ test_viterbi_ref_1
  """
  traj = simple_traj1()
  theta = np.array([1.0, 0.0])
  viterbi = TrajectoryViterbiRef(traj, theta)
  viterbi.computeMostLikely()
  assert len(viterbi.most_likely) == traj.L
  assert viterbi.most_likely[0] == 0
  assert viterbi.most_likely[1] == 0

def test_viterbi_1_1():
  """ test_viterbi_1_1
  """
  traj = simple_traj1()
  theta = np.array([1.0, 0.0])
  viterbi_ref = TrajectoryViterbiRef(traj, theta)
  viterbi_ref.computeMostLikely()
  viterbi_1 = TrajectoryViterbi1(traj, theta)
  viterbi_1.computeMostLikely()
  assert len(viterbi_1.most_likely) == traj.L
  for l in range(traj.L):
    assert viterbi_1.most_likely[l] == viterbi_ref.most_likely[l]
    assert traj.num_choices[l] == len(viterbi_ref.most_likely_tree[l])
    for i in range(traj.num_choices[l]):
      assert viterbi_ref.most_likely_tree[l][i] == \
             viterbi_1.most_likely_tree[l][i]

def test_viterbi_1_2():
  """ test_viterbi_1_2
  """
  traj = simple_traj2()
  theta = np.array([1.0, -1.0])
  viterbi_ref = TrajectoryViterbiRef(traj, theta)
  viterbi_ref.computeMostLikely()
  viterbi_1 = TrajectoryViterbi1(traj, theta)
  viterbi_1.computeMostLikely()
  assert len(viterbi_1.most_likely) == traj.L
  for l in range(traj.L):
    assert viterbi_1.most_likely[l] == viterbi_ref.most_likely[l]
    assert traj.num_choices[l] == len(viterbi_ref.most_likely_tree[l])
    for i in range(traj.num_choices[l]):
      assert viterbi_ref.most_likely_tree[l][i] == \
             viterbi_1.most_likely_tree[l][i]

def test_viterbi_1_6():
  """ test_viterbi_1_6
  """
  traj = simple_traj6()
  theta = np.array([1.0, -1.0])
  viterbi_ref = TrajectoryViterbiRef(traj, theta)
  viterbi_ref.computeMostLikely()
  viterbi_1 = TrajectoryViterbi1(traj, theta)
  viterbi_1.computeMostLikely()
  assert len(viterbi_1.most_likely) == traj.L
  for l in range(traj.L):
    assert viterbi_1.most_likely[l] == viterbi_ref.most_likely[l]
    assert traj.num_choices[l] == len(viterbi_ref.most_likely_tree[l])
    for i in range(traj.num_choices[l]):
      assert viterbi_ref.most_likely_tree[l][i] == \
             viterbi_1.most_likely_tree[l][i]

def test_viterbi_1_7():
  """ test_viterbi_1_7
  """
  traj = simple_traj7()
  theta = np.array([1.0])
  viterbi_ref = TrajectoryViterbiRef(traj, theta)
  viterbi_ref.computeMostLikely()
  viterbi_1 = TrajectoryViterbi1(traj, theta)
  viterbi_1.computeMostLikely()
  assert len(viterbi_1.most_likely) == traj.L
  for l in range(traj.L):
    assert viterbi_1.most_likely[l] == viterbi_ref.most_likely[l]
    assert traj.num_choices[l] == len(viterbi_ref.most_likely_tree[l])
    for i in range(traj.num_choices[l]):
      assert viterbi_ref.most_likely_tree[l][i] == \
             viterbi_1.most_likely_tree[l][i]
