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
''' EM algorithm to optimize the parameters.

Created on Sep 2, 2011

@author: tjhunter
'''
import numpy as np
from mm.path_inference.learning_traj_smoother import \
  TrajectorySmoother1
from mm.path_inference.learning_traj_optimizer import optimize_function

def get_traj_estims(trajs, theta):
  """ Given a set of trajectories, returns a set of trajectory estimates.
  
  i.e., a list of (traj, weights). weights is a sequence of probability
  vectors that contain the posterior probabilities from the smoother.
  """
  def inner(traj):
    """ Inner function that does the work. """
    smoother = TrajectorySmoother1(traj, theta)  
    smoother.computeProbs()
    probabilities = smoother.probabilities
    del smoother
    return (traj, probabilities)
  return [inner(traj) for traj in trajs]

def learn_em(opt_function_provider, trajs, start_value, \
             max_iters=10, max_inner_iters=5):
  """ Learn EM function.
  
  Returns the history of the progression: list of:
  - log likelihood
  - learned parameter
  
  Arguments:
  - opt_function_provider: traj_estims -> (optimizable function)
  - trajs: a set of trajectories
  - start_value: numpy array, start vector
  - max_iters: maximum number of iterations
  """
  theta = np.array(start_value)
  history = []
  for step in range(max_iters):
    traj_estims = get_traj_estims(trajs, theta)
    opt_fun = opt_function_provider(traj_estims)
    (theta1, ys) = optimize_function(opt_fun, theta, max_inner_iters)
    print 'Epoch %i : %f' % (step, ys[-1])
    history.append((ys[-1], theta1))
    theta = theta1
  return history