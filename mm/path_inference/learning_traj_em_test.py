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
''' Tests for the EM.

Created on Sep 5, 2011

@author: tjhunter
'''
from mm.path_inference.learning_traj_test import (simple_traj1, 
                                                  simple_traj4, simple_traj3)
from mm.path_inference.learning_traj_optimizer import trajs_estim_obj_fun_1
from mm.path_inference.learning_traj_em import learn_em
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')

def test_em_1():
  """ Very simple test: we pick some trajectories and verify that
  the LL increases with EM.
  """
  trajs = [simple_traj1(), simple_traj4(), simple_traj3()]
  theta_start = 0.1 * np.ones(2)
  history = learn_em(trajs_estim_obj_fun_1, trajs, theta_start)
#  (ll_end, theta_end) = history[-1]
  # Very simple check here: we verify the progression goes upward 
  # the likelihood:
  for t in range(len(history)-1):
    (ll_1, _) = history[t]
    (ll_2, _) = history[t+1]
    assert ll_1 <= ll_2
