'''
Copyright 2011, 2012 Timothy Hunter <tjhunter@eecs.berkeley.edu>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) version 3.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public 
License along with this library.  If not, see <http://www.gnu.org/licenses/>.
'''
# pylint: disable=W0105
""" Test module.  
Author: Timothy Hunter

Most of these tests correspond to implementation mistakes.
"""
from mm.path_inference.learning_traj_test import simple_traj1, \
 simple_traj2, simple_traj4, simple_traj0, simple_traj5, simple_traj8,\
  simple_traj10
from mm.path_inference.learning_traj_elements \
  import LearningElements0 as LearningElements
from mm.path_inference.learning_traj_elements \
  import LearningElements1 as LearningElementsSecure
from mm.path_inference.learning_traj_elements import LearningElementsRef
import numpy as np
# Forces Numpy to consider warnings as errors.
np.seterr(all='raise')

def within(x1, x2, epsi):
  """ Epsilon equality.
  """
  return (abs(x1-x2) < epsi)

def compute_grad(fun, x, dx=1e-8):
  """ EMpirical gradient. """
  N = len(x)
  res = np.zeros_like(x)
  fx = fun(x)
  for i in range(N):
    x2 = np.array(x)
    x2[i] += dx
    fxdx = fun(x2)
    res[i] = (fxdx - fx) / dx
  return res

def compute_hess(fun, x, dx=1e-4):
  """ Empirical hessian. """
  N = len(x)
  res = np.zeros((N, N), dtype=np.float64)
  for i in range(N):
    def fun2(y):
      """ Inner function as closure. """
      g = compute_grad(fun, y, dx)
      return g[i]
    res[i] = compute_grad(fun2, x, dx)
  return res

def test_emp_hessian_1():
  """ test_emp_hessian_1 """
  def f0(x):
    """ Inner. """
    return np.dot(x, x) / 2 
  # from mm.path_inference.learning_traj_elements_test import *
  x_0 = np.array([10.0, -100.0])
  h_ref = np.array([[1, 0], [0, 1]], dtype=np.float64)
  h = compute_hess(f0, x_0)
  assert(np.abs(h - h_ref).max() < 1e-3), (h, h_ref)

def get_fun_ref(traj, choices):
  """ Reference closure for logZ. """
  def res(theta):
    """ Inner closure. """
    elts = LearningElementsRef(traj, theta, choices)
    elts.computeLogZ()
    return elts.logZ
  return res

def get_fun_ref0(traj, choices):
  """ Reference closure for Z. """
  def res(theta):
    """ Inner closure. """
    elts = LearningElementsRef(traj, theta, choices)
    elts.computeLogZ()
    return elts.Z
  return res

def get_grad_ref(traj, choices):
  """ Reference closure for the gradient. """
  def res(theta):
    """ inner closure. """
    elts = LearningElementsRef(traj, theta, choices)
    elts.computeGradientLogZ()
    return elts.grad_logZ
  return res

def get_hess_ref(traj, choices):
  """ Reference closure for the gradient. """
  def res(theta):
    """ inner closure. """
    elts = LearningElementsRef(traj, theta, choices)
    elts.computeHessianLogZ()
    return elts.hess_logZ
  return res

def get_hess_ref0(traj, choices):
  """ Reference closure for the hessian of Z. """
  def res(theta):
    """ inner closure. """
    elts = LearningElementsRef(traj, theta, choices)
    elts.computeHessianLogZ()
    return elts.hess_Z
  return res

def test_gradient_ref1():
  """ Simple test to check that the reference implementation of the gradient
  is equal to a empirical definition based on logZ.
  """
  traj = simple_traj1()
  choices = [1, 0]
  fun = get_fun_ref(traj, choices)
  grad_fun = get_grad_ref(traj, choices)
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([0.0, 0.0])
  g_emp = compute_grad(fun, theta)
  g = grad_fun(theta)
  assert(np.abs(g - g_emp).max() < 1e-3), (g, g_emp)

def test_gradient_ref2():
  """ Test at a different point. """
  traj = simple_traj1()
  choices = [1, 0]
  fun = get_fun_ref(traj, choices)
  grad_fun = get_grad_ref(traj, choices)
  theta = np.array([1.0, -1.0])
  g_emp = compute_grad(fun, theta)
  g = grad_fun(theta)
  assert(np.abs(g - g_emp).max() < 1e-3), (g, g_emp)

def test_gradient_ref_traj4():
  """ test_gradient_ref_traj4
   Simple test to check that the reference implementation of the gradient
  is equal to a empirical definition based on logZ.
  """
  traj = simple_traj4()
  choices = [1, 0, 2]
  fun = get_fun_ref(traj, choices)
  grad_fun = get_grad_ref(traj, choices)
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([0.0, 0.0])
  g_emp = compute_grad(fun, theta)
  g = grad_fun(theta)
  assert(np.abs(g - g_emp).max() < 1e-3), (g, g_emp)

def test_gradient_ref_traj0_1():
  """ test_gradient_ref_traj0_1. """
  traj = simple_traj0()
  choices = [1, 0]
  fun = get_fun_ref(traj, choices)
  grad_fun = get_grad_ref(traj, choices)
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([1.0])
  g_emp = compute_grad(fun, theta)
  g = grad_fun(theta)
  assert(np.abs(g - g_emp).max() < 1e-3), (g, g_emp)

def test_gradient_ref_traj5_1():
  """ test_gradient_ref_traj5_1. """
  traj = simple_traj5()
  choices = [1, 0, 2]
  fun = get_fun_ref(traj, choices)
  grad_fun = get_grad_ref(traj, choices)
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([1.0])
  g_emp = compute_grad(fun, theta)
  g = grad_fun(theta)
  assert(np.abs(g - g_emp).max() < 1e-3), (g, g_emp)


def test_hessian_ref1():
  """ Simple test to check that the reference implementation of the hessian
  is equal to a empirical definition based on logZ.
  """
  traj = simple_traj1()
  choices = [1, 0]
  fun = get_fun_ref(traj, choices)
  hess_fun = get_hess_ref(traj, choices)
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([0.0, 0.0])
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-3), (h, h_emp)

def test_hessian_ref2():
  """ Simple test to check that the reference implementation of the hessian
  is equal to a empirical definition based on logZ.
  """
  traj = simple_traj1()
  choices = [1, 0]
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([1.0, -1.0])
  # Z
  fun = get_fun_ref0(traj, choices)
  hess_fun = get_hess_ref0(traj, choices)
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-3), (h, h_emp)
  # LogZ
  fun = get_fun_ref(traj, choices)
  hess_fun = get_hess_ref(traj, choices)
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-3), (h, h_emp)

def test_hessian_ref3():
  """ Simple test to check that the reference implementation of the hessian
  is equal to a empirical definition based on logZ.
  """
  traj = simple_traj1()
  choices = [0, 0]
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([-0.1, 0.1])
  # Z
  fun = get_fun_ref0(traj, choices)
  hess_fun = get_hess_ref0(traj, choices)
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-3), (h, h_emp)
  # LogZ
  fun = get_fun_ref(traj, choices)
  hess_fun = get_hess_ref(traj, choices)
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-3), (h, h_emp)

def test_hessian_ref_traj5_1():
  """ test_hessian_ref_traj5_1
  
  Simple test to check that the reference implementation of the hessian
  is equal to a empirical definition based on logZ.
  """
  traj = simple_traj5()
  choices = [1, 0, 2]
  # Do not forget to define the vector as floating point numbers!!!
  theta = np.array([1.0])
  # Z
  fun = get_fun_ref0(traj, choices)
  hess_fun = get_hess_ref0(traj, choices)
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-2), (h, h_emp)
  # LogZ
  fun = get_fun_ref(traj, choices)
  hess_fun = get_hess_ref(traj, choices)
  h_emp = compute_hess(fun, theta)
  h = hess_fun(theta)
  assert(np.abs(h - h_emp).max() < 1e-3), (h, h_emp)

def test_Z1():
  """ Test of implementation 1. """
  traj = simple_traj1()
  theta = np.array([0.0, -1.0])
  choices = [1, 0]
  elts = LearningElements(traj, theta, choices)
  elts.computeLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeLogZ()
  assert(within(elts.logZ, elts_ref.logZ, 1e-5))

def test_traj_5_1():
  """ test_traj_5_1 """
  traj = simple_traj5()
  theta = np.array([-1.0])
  choices = [1, 0, 2]
  elts = LearningElements(traj, theta, choices)
  elts.computeLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeLogZ()
  assert(within(elts.Z, elts_ref.Z, 1e-5)), (elts.Z, elts_ref.Z, 1e-5)
  assert(within(elts.logZ, elts_ref.logZ, 1e-5)), \
    (elts.logZ, elts_ref.logZ, 1e-5)

def test_traj_5_2():
  """ test_traj_5_2 """
  traj = simple_traj5()
  theta = np.array([-1.0])
  choices = [1, 0, 2]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeLogZ()
  assert(within(elts.Z, elts_ref.Z, 1e-5)), (elts.Z, elts_ref.Z, 1e-5)
  assert(within(elts.logZ, elts_ref.logZ, 1e-5)), \
    (elts.logZ, elts_ref.logZ, 1e-5)

def test_traj_1_2():
  """ test_traj_1_2 """
  traj = simple_traj1()
  theta = np.array([1.0, -1.0])
  choices = [1, 0]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeLogZ()
  assert(within(elts.Z, elts_ref.Z, 1e-5)), (elts.Z, elts_ref.Z, 1e-5)
  assert(within(elts.logZ, elts_ref.logZ, 1e-5)), \
    (elts.logZ, elts_ref.logZ, 1e-5)

def test_grad_traj1_1():
  """ test_grad_traj1_1
  Test of implementation 1 of gradient. """
  traj = simple_traj1()
  theta = np.array([0.0, -1.0])
  choices = [1, 0]
  elts = LearningElements(traj, theta, choices)
  elts.computeGradientLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeGradientLogZ()
  g = elts.grad_logZ
  g_ref = elts_ref.grad_logZ
  assert(np.abs(g - g_ref).max() < 1e-3), (g, g_ref)

def test_grad_traj1_2():
  """ test_grad_traj1_2
  Test of implementation 1 of gradient. """
  traj = simple_traj1()
  theta = np.array([0.0, -1.0])
  choices = [1, 0]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeGradientLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeGradientLogZ()
  g = elts.grad_logZ
  g_ref = elts_ref.grad_logZ
  assert(np.abs(g - g_ref).max() < 1e-3), (g, g_ref)

def test_grad_traj5_2():
  """ test_grad_traj5_2
  Test of implementation 1 of gradient. """
  traj = simple_traj5()
  theta = np.array([-1.0])
  choices = [1, 0, 2]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeGradientLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeGradientLogZ()
  g = elts.grad_logZ
  g_ref = elts_ref.grad_logZ
  assert(np.abs(g - g_ref).max() < 1e-3), (g, g_ref)

def test_hess_traj1_1():
  """ test_hess_traj1_1 """
  traj = simple_traj1()
  theta = np.array([0.0, -1.0])
  choices = [1, 0]
  elts = LearningElements(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), (h, h_ref)

def test_hess_traj1_2():
  """ test_hess_traj1_2 """
  traj = simple_traj1()
  theta = np.array([0.0, -1.0])
  choices = [1, 0]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), (h, h_ref)

def test_hess_traj2_1():
  """ test_hess_traj1_1 """
  traj = simple_traj2()
  theta = np.array([0.0, -1.0])
  choices = [0, 1]
  elts = LearningElements(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), (h, h_ref)

def test_hess_traj4_1():
  """ test_hess_traj4_1 """
  traj = simple_traj4()
  theta = np.array([0.0, -1.0])
  choices = [1, 0, 2]
  elts = LearningElements(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), (h, h_ref)

def test_hess_traj5_1():
  """ test_hess_traj5_1 """
  traj = simple_traj5()
  theta = np.array([-1.0])
  choices = [1, 0, 2]
  elts = LearningElements(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), \
    (h, h_ref, elts.hess_Z, elts_ref.hess_Z, \
     elts.grad_Z, elts_ref.grad_Z, elts.Z, elts_ref.Z,)

def test_hess_traj5_2():
  """ test_hess_traj5_2 """
  traj = simple_traj5()
  theta = np.array([-1.0])
  choices = [1, 0, 2]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), \
    (h, h_ref, elts.hess_Z, elts_ref.hess_Z, \
     elts.grad_Z, elts_ref.grad_Z, elts.Z, elts_ref.Z,)

def test_hess_traj8_2():
  """ test_hess_traj8_2 """
  traj = simple_traj8()
  theta = np.array([0.0])
  choices = [1, 0]
  elts = LearningElementsSecure(traj, theta, choices)
  elts.computeHessianLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeHessianLogZ()
  h = elts.hess_logZ
  h_ref = elts_ref.hess_logZ
  assert(np.abs(h - h_ref).max() < 1e-3), \
    (h, h_ref, elts.hess_Z, elts_ref.hess_Z, \
     elts.grad_Z, elts_ref.grad_Z, elts.Z, elts_ref.Z,)

def test_Z2():
  """ Test of implementation 2. """
  traj = simple_traj2()
  theta = np.array([0.0, -1.0])
  choices = [0, 1]
  elts = LearningElements(traj, theta, choices)
  elts.computeLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeLogZ()
  assert(within(elts.logZ, elts_ref.logZ, 1e-5))

def test_Z3():
  """ Test of implementation 2. """
  traj = simple_traj10()
  theta = np.array([500.0, -100.0])
  weights = [[0.1, 0.9], [0.9, 0.1]]
  elts = LearningElementsSecure(traj, theta, weights=weights)
  elts.computeLogZ()

def test_grad_Z2():
  """ Test of implementation 2 of gradient. """
  traj = simple_traj2()
  theta = np.array([0.0, -1.0])
  choices = [0, 1]
  elts = LearningElements(traj, theta, choices)
  elts.computeGradientLogZ()
  elts_ref = LearningElementsRef(traj, theta, choices)
  elts_ref.computeGradientLogZ()
  g = elts.grad_logZ
  g_ref = elts_ref.grad_logZ
  assert(np.abs(g - g_ref).max() < 1e-3), (g, g_ref)

if __name__ == '__main__':
  def f(x):
    """ Inner. """
    return np.dot(x, x) / 2 
  # from mm.path_inference.learning_traj_elements_test import *
  x0 = np.array([10.0, -100.0])
  compute_hess(f, x0)