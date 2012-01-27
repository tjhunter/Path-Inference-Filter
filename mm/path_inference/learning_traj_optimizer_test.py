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
""" Optim test.
"""

#from mm.path_inference.learning_traj_test import simple_traj1,\
# simple_traj2, simple_traj4, simple_traj0, simple_traj5

from mm.path_inference.learning_traj_elements_test \
  import compute_grad, compute_hess, within
from mm.path_inference.learning_traj_optimizer import optimize_function, \
  trajs_obj_fun_0, trajs_obj_fun_1, trajs_obj_fun_ref, \
  trajs_estim_obj_fun_ref, trajs_obj_fun_0bis, trajs_obj_fun_2
from mm.path_inference.learning_traj_test import \
 simple_traj10, simple_traj4, simple_traj9, simple_traj5, simple_traj8
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')

def wrap(fun):
  """ Wrapper that uses empirical functions only.
  """
  def inner(x):
    """ Closure. """
    y = fun(x)
    g = compute_grad(fun, x)
    h = compute_hess(fun, x)
    return (y, g, h)
  return inner

def fun1(x):
  """ Simple nearly quadratic function.
  """
  return -pow(np.dot(x - np.ones_like(x), x - np.ones_like(x)), 0.7)

def test_1():
  """ Basic test for the optimizer. """
  start = np.zeros(2)
  opt_fun = wrap(fun1)
  (x, ys) = optimize_function(opt_fun, start, 10)
  assert(np.abs(x-np.ones_like(x)).max() < 1e-3)
  assert ys[-1] >= -1e-3
  assert ys[-1] <= 0
  assert np.abs(start).max() == 0, (start, x)

def test_opt_traj5_1ref():
  """ test_opt_traj5_1 """
  traj = simple_traj5()
  start_theta = np.array([-1.0])
  choices = [1, 0, 2]
  traj_choices = [(traj, choices)]
  obj_fun = trajs_obj_fun_ref(traj_choices)
  (theta, ys) = optimize_function(obj_fun, start_theta, max_iters=20)
  assert ys[0] <= ys[-1], (theta, ys, len(ys))

def test_opt_traj4_1ref():
  """ test_opt_traj4_1 """
  traj = simple_traj4()
  start_theta = np.array([0.0, 0.0])
  choices = [1, 0, 2]
  traj_choices = [(traj, choices)]
  obj_fun = trajs_obj_fun_ref(traj_choices)
  (theta, ys) = optimize_function(obj_fun, start_theta, max_iters=20)
  assert ys[0] <= ys[-1], (theta, ys, len(ys))

def test_opt0_traj4_1ref():
  """ test_opt0_traj4_1 """
  traj = simple_traj4()
  start_theta = np.array([0.0, 0.0])
  choices = [1, 0, 2]
  estims = [np.array([0.0, 1.0]), np.array([1.0]), np.array([0.0, 0.0, 1.0])]
  traj_choices = [(traj, choices)]
  traj_estims = [(traj, estims)]
  obj_fun_2 = trajs_obj_fun_2(traj_choices)
  obj_fun_1 = trajs_obj_fun_1(traj_choices)
  obj_fun_0 = trajs_obj_fun_0(traj_choices)
  obj_fun_ref = trajs_obj_fun_ref(traj_choices)
  obj_fun_estim_ref = trajs_estim_obj_fun_ref(traj_estims)
  max_iters = 5
  (theta_0, ys_0) = optimize_function(obj_fun_0, start_theta, max_iters)
  (theta_1, ys_1) = optimize_function(obj_fun_1, start_theta, max_iters)
  (theta_2, ys_2) = optimize_function(obj_fun_2, start_theta, max_iters)
  (theta_ref, ys_ref) = optimize_function(obj_fun_ref, start_theta, max_iters)
  (theta_estim_ref, ys_estim_ref) = optimize_function(obj_fun_estim_ref, \
                                                      start_theta, max_iters)
  assert(np.abs(theta_0 - theta_ref).max() < 1e-3), (theta_0, theta_ref)
  assert(np.abs(theta_1 - theta_ref).max() < 1e-3), (theta_1, theta_ref)
  assert(np.abs(theta_2 - theta_ref).max() < 1e-3), (theta_2, theta_ref)
  assert(np.abs(theta_estim_ref - theta_ref).max() < 1e-3), (theta_estim_ref, \
                                                             theta_ref)
  assert within(ys_0[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_0, ys_0)
  assert within(ys_1[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_1, ys_1)
  assert within(ys_2[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_2, ys_2)
  assert within(ys_estim_ref[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, \
                                                    theta_estim_ref, \
                                                    ys_estim_ref)

def test_opt0_traj5_1():
  """ test_opt0_traj5_1 """
  traj = simple_traj5()
  start_theta = np.array([-1.0])
  choices = [1, 0, 2]
  traj_choices = [(traj, choices)]
  obj_fun_1 = trajs_obj_fun_1(traj_choices)
  obj_fun_0 = trajs_obj_fun_0(traj_choices)
  obj_fun_ref = trajs_obj_fun_ref(traj_choices)
  max_iters = 5
  (theta_0, ys_0) = optimize_function(obj_fun_0, start_theta, max_iters)
  (theta_1, ys_1) = optimize_function(obj_fun_1, start_theta, max_iters)
  (theta_ref, ys_ref) = optimize_function(obj_fun_ref, start_theta, max_iters)
  assert(np.abs(theta_0 - theta_ref).max() < 1e-3), (theta_0, theta_ref)
  assert(np.abs(theta_1 - theta_ref).max() < 1e-3), (theta_1, theta_ref)
  assert within(ys_0[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_0, ys_0)
  assert within(ys_1[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_1, ys_1)

def test_opt0_traj8_1():
  """ test_opt0_traj8_1 """
  traj = simple_traj8()
  start_theta = np.array([-1.0])
  choices = [1, 0]
  traj_choices = [(traj, choices)]
  obj_fun_1 = trajs_obj_fun_1(traj_choices)
  obj_fun_0 = trajs_obj_fun_0(traj_choices)
  obj_fun_0bis = trajs_obj_fun_0bis(traj_choices)
  obj_fun_ref = trajs_obj_fun_ref(traj_choices)
  max_iters = 5
  (theta_ref, ys_ref) = optimize_function(obj_fun_ref, start_theta, \
                                          max_iters, regularizer=1e-4)
  print ys_ref, theta_ref
  assert np.abs(theta_ref) < 1e-4
  (theta_0bis, ys_0bis) = optimize_function(obj_fun_0bis, \
                                            start_theta, max_iters, \
                                            regularizer=1e-4)
  (theta_0, ys_0) = optimize_function(obj_fun_0, \
                                      start_theta, max_iters, regularizer=1e-4)
  (theta_1, ys_1) = optimize_function(obj_fun_1, \
                                      start_theta, max_iters, regularizer=1e-4)
  assert(np.abs(theta_0bis - theta_ref).max() < 1e-3), (theta_0bis, theta_ref)
  assert(np.abs(theta_0 - theta_ref).max() < 1e-3), (theta_0, theta_ref)
  assert(np.abs(theta_1 - theta_ref).max() < 1e-3), (theta_1, theta_ref)
  assert within(ys_0bis[0], ys_ref[0], 1e-3), \
    (theta_ref, ys_ref, theta_0bis, ys_0bis)
  assert within(ys_0[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_0, ys_0)
  assert within(ys_1[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_1, ys_1)

def test_opt0_traj9_1():
  """ test_opt0_traj9_1 """
  traj = simple_traj9()
  start_theta = np.array([-1.0])
  choices = [0, 0]
  traj_choices = [(traj, choices)]
  obj_fun_2 = trajs_obj_fun_2(traj_choices)
  obj_fun_1 = trajs_obj_fun_1(traj_choices)
  obj_fun_0 = trajs_obj_fun_0(traj_choices)
  obj_fun_0bis = trajs_obj_fun_0bis(traj_choices)
  obj_fun_ref = trajs_obj_fun_ref(traj_choices)
  max_iters = 5
  (theta_ref, ys_ref) = \
    optimize_function(obj_fun_ref, start_theta, max_iters, regularizer=1e-4)
  print ys_ref, theta_ref
  (theta_0bis, ys_0bis) = \
    optimize_function(obj_fun_0bis, start_theta, max_iters, regularizer=1e-4)
  (theta_0, ys_0) = \
    optimize_function(obj_fun_0, start_theta, max_iters, regularizer=1e-4)
  (theta_1, ys_1) = \
    optimize_function(obj_fun_1, start_theta, max_iters, regularizer=1e-4)
  (theta_2, ys_2) = \
    optimize_function(obj_fun_2, start_theta, max_iters, regularizer=1e-4)
  assert(np.abs(theta_0bis - theta_ref).max() < 1e-3), (theta_0bis, theta_ref)
  assert(np.abs(theta_0 - theta_ref).max() < 1e-3), (theta_0, theta_ref)
  assert(np.abs(theta_1 - theta_ref).max() < 1e-3), (theta_1, theta_ref)
  assert(np.abs(theta_2 - theta_ref).max() < 1e-3), (theta_2, theta_ref)
  assert within(ys_0bis[0], ys_ref[0], 1e-3), \
    (theta_ref, ys_ref, theta_0bis, ys_0bis)
  assert within(ys_0[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_0, ys_0)
  assert within(ys_1[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_1, ys_1)
  assert within(ys_2[0], ys_ref[0], 1e-3), (theta_ref, ys_ref, theta_2, ys_2)

def test_opt0_traj10_1():
  """ test_opt0_traj10_1 """
  traj = simple_traj10()
  start_theta = np.array([-1.0, 2.0])
  choices = [0, 0]
  traj_choices = [(traj, choices)]
  obj_fun_2 = trajs_obj_fun_2(traj_choices)
  obj_fun_1 = trajs_obj_fun_1(traj_choices)
  obj_fun_0bis = trajs_obj_fun_0bis(traj_choices)
  max_iters = 5
  (theta_0bis, ys_0bis) = \
    optimize_function(obj_fun_0bis, start_theta, max_iters, regularizer=1e-4)
  (theta_1, ys_1) = \
    optimize_function(obj_fun_1, start_theta, max_iters, regularizer=1e-4)
  (theta_2, ys_2) = \
    optimize_function(obj_fun_2, start_theta, max_iters, regularizer=1e-4)
  assert(np.abs(theta_1 - theta_0bis).max() < 1e-3), (theta_1, theta_0bis)
  assert(np.abs(theta_2 - theta_0bis).max() < 1e-3), (theta_2, theta_0bis)
  assert within(ys_1[0], ys_0bis[0], 1e-3), (theta_0bis, ys_0bis, theta_1, ys_1)
  assert within(ys_2[0], ys_0bis[0], 1e-3), (theta_0bis, ys_0bis, theta_2, ys_2)
