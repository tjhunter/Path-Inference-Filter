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
""" Set of functions for computing the most likely parameter vector.
"""

from numpy.linalg import inv
from numpy import dot
import numpy as np
from mm.path_inference.learning_traj_elements import LearningElementsRef, \
                                                      LearningElements0, \
                                                      LearningElements1, \
                                                      LearningElements0Bis, \
                                                      LearningElements2

def optimize_function(fun, start_point, max_iters=5, regularizer=0):
  """ Optimization procedure for a function.
  
  Arguments:
   - fun: function to optimize: 1-nparray -> (y, grad, hessian)
   - start_point: 1-nparray
   - max_iters: the maximum number of iterations
  Returns:
   (x, ys)
  where x is the best value and and ys is the list of evaluated values
  of the function.
  """
  ys = []
  x = start_point.copy()
  n = len(x)
  d = regularizer * np.eye(n) / 2
  while len(ys) < max_iters:
    print 'iter %s, x=' % len(ys), x
    (y, g, h) = fun(x)
    y += - dot(x, dot(d, x))
    g += - 2 * dot(d, x)
    h += - 2 * d
    ys.append(y)
    print 'y=',y
    search_dir = - dot(inv(h), g)
    t = 1
    alpha = 0.2
    beta = 0.7
#    print "LINE SEARCH"
    while True:
      x2 = x + t * search_dir
#      print 'Looking t = ',t
      (y_, _, _) = fun(x2)
#      print 'Y_ = ',y_
      if y_ >= y + alpha * t * dot(g, search_dir):
        break
      else:
        t *= beta
      if t < 1e-5:
        print 't is too small'
        break
    x = x2
  return (x, ys)  

def trajs_obj_fun_ref(traj_choices):
  """ Uses the reference implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, choice) in traj_choices:
      elts = LearningElementsRef(traj, theta, choice)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner

def trajs_obj_fun_0(traj_choices):
  """ Uses the reference implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, choice) in traj_choices:
      elts = LearningElements0(traj, theta, choice)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner

def trajs_obj_fun_0bis(traj_choices):
  """ Uses the reference implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, choice) in traj_choices:
      elts = LearningElements0Bis(traj, theta, choice)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner

def trajs_obj_fun_1(traj_choices):
  """ Uses the rsecure implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, choice) in traj_choices:
      elts = LearningElements1(traj, theta, choice)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner

def trajs_obj_fun_2(traj_choices):
  """ Uses the rsecure implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, choice) in traj_choices:
      elts = LearningElements2(traj, theta, choice)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner

def trajs_estim_obj_fun_ref(traj_estims):
  """ Uses the reference implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, weights) in traj_estims:
      elts = LearningElementsRef(traj, theta, weights=weights)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner

def trajs_estim_obj_fun_1(traj_estims):
  """ Uses the reference implementation to create an objective function.
  """
  def inner(theta):
    """ Returned closure. """
    y = 0.0
    g = np.zeros_like(theta)
    n = len(theta)
    h = np.zeros((n, n))
    for (traj, weights) in traj_estims:
      elts = LearningElements1(traj, theta, weights=weights)
      elts.computeValue()
      y += elts.logValue
      elts.computeGradientValue()
      g += elts.grad_logValue
      elts.computeHessianValue()
      h += elts.hess_logValue
    return (y, g, h)
  return inner
