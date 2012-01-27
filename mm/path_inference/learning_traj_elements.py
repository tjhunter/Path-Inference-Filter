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
""" Computation elements for the learning trajectory, simple version.

This class is not optimized with respect to log underflows.
"""
import numpy as np
from numpy import dot, exp, log, outer
from mm.log_math import lse, lse_vec, MINUS_INF, lse_npy, lse_vec_npy, lse_vec_2

def check_weight(weight):
  """ Checks the weight vector respects a number of assumptions. """
  assert(len(weight.shape) == 1)
  assert(weight.sum() > 0.99)
  assert(weight.sum() < 1.01)
  assert((weight >= 0).all())
  assert((weight <= 1).all())

class LearningElementsRef:
  """ Elements for computing value, gradients, etc.
  
  This is the reference implementation (slow but correct, for testing).
  This implementation does not handle large feature values.
  
  Useful fields:
  - traj: a LearningTrajectory object.
  - theta: the weight vector
  - lZ: the logarithm of Z
  - gradLZ
  - hessLZ
  """
  # pylint: disable=W0201
  def __init__(self, learning_traj, theta, choices=None, weights=None):
    self.traj = learning_traj
    self.theta = np.array(theta)
    # Computing and checkin the weight vectors:
    assert(choices or weights)
    if weights:
      assert(self.traj.L == len(weights))
      self.weights = [np.array(w, dtype=np.double) for w in weights]
    if choices:
      assert(self.traj.L == len(choices)), \
        (self.traj.L, len(choices))
      self.weights = [np.zeros([Nl]) for Nl in self.traj.num_choices]
      for l in range(self.traj.L):
        c = choices[l]
        assert c >= 0, (l, choices[l])
        assert c < self.traj.num_choices[l], (l, choices[l], 
                                              self.traj.num_choices[l])
        self.weights[l][c] = 1.0
    self.checkAssertions()

  def checkAssertions(self):
    """ Tests if a number of invariants are respected.
    """
    # Check theta
    assert(len(self.theta.shape) == 1)
    N = self.theta.shape[0]
    L = self.traj.L
    for l in range(L):
      weight = self.weights[l]
#      weight[weight<0.1] = 0
#      weight[weight>=0.1] = 1
      assert(abs(np.sum(weight)-1)<1e-5), weight
      assert(np.all(weight>=0)), weight
      feats = self.traj.features_np[l]
      assert(len(feats.shape) == 2)
      N_l = self.traj.num_choices[l]
      assert(feats.shape[0] == N_l)
      assert(feats.shape[1] == N)
      assert(weight.shape[0] == N_l)
  
  def computeSStats(self):
    """ Computes sufficient statistics. """
    if 'sstats' in self.__dict__:
      return
    L = self.traj.L
    # Expected sufficient statistics:
    self.sstats = [dot(self.weights[l], self.traj.features_np[l]) \
                   for l in range(L)]
  
  def computeValue(self):
    """
      Calls LogZ.
      Provides logValue
    """
    if 'logValue' in self.__dict__:
      return
    self.computeSStats()
    self.computeLogZ()
    L = self.traj.L
    # Expected sufficient statistics:
    self.logV1 = sum([dot(self.sstats[l], self.theta) for l in range(L)])
    assert self.logV1 <= self.logZ, (self.logV1, self.logZ, self.sstats)
    self.logValue = self.logV1 - self.logZ
  
  def computeGradientValue(self):
    """
      Calls LogZ.
      Provides logValue
    """
    if 'grad_logValue' in self.__dict__:
      return
    self.computeSStats()
    L = self.traj.L
    # Expected sufficient statistics:
    self.grad_logV1 = sum([self.sstats[l] for l in range(L)])
    self.computeGradientLogZ()
    self.grad_logValue = self.grad_logV1 - self.grad_logZ
  
  def computeHessianValue(self):
    """
      Calls LogZ.
      Provides logValue
    """
    if 'hess_logValue' in self.__dict__:
      return
    # Expected sufficient statistics:
    self.computeHessianLogZ()
    self.hess_logValue = - self.hess_logZ

  def computeLogZ(self):
    """ Reference computation of Z and logZ. Very inefficient but correct.
    Provides:
      - logZ
    Debug info:
      - Z
    """
    if 'logZ' in self.__dict__:
      return
    L = self.traj.L
    # Weighting statistics.
    self.Z = 0
    def inner(indices):
      """ Inner working function. """
      L_ = len(indices)
      if L_ == L:
        v1 = np.zeros_like(self.theta)
        for l in range(L):
          v1 += self.traj.features_np[l][indices[l]]
        self.Z += exp(dot(v1, self.theta))
      else:
        i = indices[-1]
        if i in self.traj.connections_forward[L_-1]:
          for j in self.traj.connections_forward[L_-1][i]:
            inner(indices + [j])
    for i in range(self.traj.num_choices[0]):
      inner([i])
    self.logZ = log(self.Z)

  def computeGradientLogZ(self):
    """ Computes the gradient of the logarithm of Z.
    """
    if 'grad_logZ' in self.__dict__:
      return
    self.computeLogZ()
    self.grad_Z = np.zeros_like(self.theta)
    L = self.traj.L
    def inner(indices):
      """ Inner work function. """
      L_ = len(indices)
      if L_ == L:
        v1 = np.zeros_like(self.theta)
        for l in range(L):
          v1 += self.traj.features_np[l][indices[l]]
        g = exp(dot(v1, self.theta)) * v1
        # print(v1, self.theta, dot(v1, self.theta), g)
        self.grad_Z += g
      else:
        i = indices[-1]
        if i in self.traj.connections_forward[L_-1]:
          for j in self.traj.connections_forward[L_-1][i]:
            inner(indices + [j])
    for i in range(self.traj.num_choices[0]):
      inner([i])
    # print('Z=', self.Z)
    # print('grad_Z=', self.grad_Z)
    self.grad_logZ = self.grad_Z / self.Z

  def computeHessianLogZ(self):
    """ Hessian of log(Z). """
    if 'hess_logZ' in self.__dict__:
      return
    N = len(self.theta)
    L = self.traj.L
    self.computeLogZ()
    self.computeGradientLogZ()
    self.hess_Z = np.zeros((N, N))
    # Inner recursive function.
    # The stack should be large enough for our purpose.
    def inner(indices):
      """ Inner working function. """
      L_ = len(indices)
      if L_ == L:
        v1 = np.zeros_like(self.theta)
        for l in range(L):
          v1 += self.traj.features_np[l][indices[l]]
        h = exp(dot(v1, self.theta)) * outer(v1, v1)
        self.hess_Z += h
      else:
        i = indices[-1]
        if i in self.traj.connections_forward[L_-1]:
          for j in self.traj.connections_forward[L_-1][i]:
            inner(indices + [j])
    for i in range(self.traj.num_choices[0]):
      inner([i])
    self.hess_logZ = self.hess_Z / self.Z \
                     - outer(self.grad_Z, self.grad_Z) / (self.Z * self.Z)

class LearningElements0Bis(LearningElementsRef):
  """ Another test implementation that is secure against underflows (i.e., 
  large feature vectors) but is exponentially slow to compute.
  Can be used on production data for small trajectories to make sure the 
  optimization procedures are correct.
  """
  def computeLogZ(self):
    """ Reference computation of Z and logZ, in the log domain. 
    Very inefficient.
    Provides:
      - logZ
    Debug info:
      - Z
    """
    if 'logZ' in self.__dict__:
      return
    L = self.traj.L
    # Weighting statistics.
    self.Z = 0
    self.logZ = MINUS_INF
    # List of the log weights of all the possible trajectories.
    intermediate_log_vals = []
    def inner(indices):
      """ Inner working function. """
      L_ = len(indices)
      if L_ == L:
        v1 = np.zeros_like(self.theta)
        for l in range(L):
          v1 += self.traj.features_np[l][indices[l]]
        intermediate_log_vals.append(dot(v1, self.theta))
      else:
        i = indices[-1]
        for j in self.traj.connections_forward[L_-1][i]:
          inner(indices + [j])
    for i in range(self.traj.num_choices[0]):
      inner([i])
    self.logZ = lse(intermediate_log_vals)
    # Make sure we do not overflow.
    if self.logZ < 100 and self.logZ > -100:
      self.Z = exp(self.logZ)

  # pylint: disable=W0201
  def computeGradientLogZ(self):
    """ Computes the gradient of the logarithm of Z.
    """
    if 'grad_logZ' in self.__dict__:
      return
    self.computeLogZ()
    self.grad_Z = np.zeros_like(self.theta)
    # List of the log weighted gradients of all the possible trajectories.
    intermediate_log_grads = []
    L = self.traj.L
    def inner(indices):
      """ Inner work function. """
      L_ = len(indices)
      if L_ == L:
        v1 = np.zeros_like(self.theta)
        for l in range(L):
          v1 += self.traj.features_np[l][indices[l]]
        log_g = (dot(v1, self.theta), v1)
        # print(v1, self.theta, dot(v1, self.theta), g)
        intermediate_log_grads.append(log_g)
      else:
        i = indices[-1]
        if i in self.traj.connections_forward[L_-1]:
          for j in self.traj.connections_forward[L_-1][i]:
            inner(indices + [j])
    for i in range(self.traj.num_choices[0]):
      inner([i])
    (log_scale, v) = lse_vec(intermediate_log_grads)
    self.grad_logZ = exp(log_scale - self.logZ) * v
    self.log_gradZ = (log_scale, v)

  # pylint: disable=W0201
  def computeHessianLogZ(self):
    """ Hessian of log(Z). """
    if 'hess_logZ' in self.__dict__:
      return
    L = self.traj.L
    self.computeLogZ()
    self.computeGradientLogZ()
    intermediate_log_hessians = []
    # Inner recursive function.
    # The stack should be large enough for our purpose.
    def inner(indices):
      """ Inner working function. """
      L_ = len(indices)
      if L_ == L:
        v1 = np.zeros_like(self.theta)
        for l in range(L):
          v1 += self.traj.features_np[l][indices[l]]
        h = (dot(v1, self.theta), outer(v1, v1))
        intermediate_log_hessians.append(h)
      else:
        i = indices[-1]
        if i in self.traj.connections_forward[L_-1]:
          for j in self.traj.connections_forward[L_-1][i]:
            inner(indices + [j])
    for i in range(self.traj.num_choices[0]):
      inner([i])
    self.log_hess_Z = lse_vec(intermediate_log_hessians)
    (grad_log_scale, grad_vec) = self.log_gradZ
    (hess_log_scale, hess_vec) = self.log_hess_Z
    (lhess_log_scale, lhess_vec) = \
      lse_vec([(hess_log_scale - self.logZ, hess_vec), 
               (2 * grad_log_scale - 2 * self.logZ, -outer(grad_vec, 
                                                           grad_vec))])
    self.hess_logZ = exp(lhess_log_scale) * lhess_vec

class LearningElements0(LearningElementsRef):
  """ Naive implementation for computing the elements.
  
  This implementation is much faster than the reference implementation (linear
  as opposed to exponential) but is not safe against log underflows.
  
  Useful fields:
  - traj: a LearningTrajectory object.
  - theta: the weight vector
  - ...
  """    
  # pylint: disable=W0201
  def computeLogZ(self):
    """
    Provides Zs, logZ
    """
    if 'logZ' in self.__dict__:
      return
    L = self.traj.L
    # Sufficient statistics before the weight.
    self.w_sstats = [exp(dot(self.traj.features_np[l], self.theta)) \
                     for l in range(L)]
    self.Zs = [self.w_sstats[0]]
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      vec = np.zeros((N_l))
      conns_back = self.traj.connections_backward[l]
      for i in range(N_l):
        if i in conns_back:
          for j in conns_back[i]:
            vec[i] += self.w_sstats[l][i] * self.Zs[l-1][j]
      self.Zs.append(vec)
    assert(len(self.Zs) == L)
    self.Z = sum(self.Zs[L-1])
    self.logZ = log(self.Z)

  def computeGradientLogZ(self):
    if 'grad_logZ' in self.__dict__:
      return
    self.computeLogZ()
    N = len(self.theta)
    L = self.traj.L
    # The initial values:
    N_0 = self.traj.num_choices[0]
    grad_Zs0 = np.zeros((N_0, N))
    for i in range(N_0):
      T_i_0 = self.traj.features_np[0][i]
      grad_Zs0[i] += exp(dot(T_i_0, self.theta)) * T_i_0
    self.grad_Zs = [grad_Zs0]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      vec = np.zeros((N_l, N))
      conns_back = self.traj.connections_backward[l]
      for i in range(N_l):
        vec[i] += self.Zs[l][i] * self.traj.features_np[l][i]
        if i in conns_back:
          for j in conns_back[i]:
            vec[i] += self.w_sstats[l][i] * self.grad_Zs[l-1][j]
      self.grad_Zs.append(vec)
    assert(len(self.grad_Zs) == L)
    self.grad_Z = sum(self.grad_Zs[L-1])
    self.grad_logZ = self.grad_Z / self.Z

  def computeHessianLogZ(self):
    """ Hessian of log(Z). """
    if 'hess_logZ' in self.__dict__:
      return
    self.computeLogZ()
    self.computeGradientLogZ()
    N = len(self.theta)
    L = self.traj.L
    # The initial values:
    N_0 = self.traj.num_choices[0]
    hess_Zs0 = [np.zeros((N, N)) for i in range(N_0)]
    for i in range(N_0):
      T_i_0 = self.traj.features_np[0][i]
      hess_Zs0[i] = exp(dot(T_i_0, self.theta)) * outer(T_i_0, T_i_0)
    self.hess_Zs = [hess_Zs0]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      vec = [np.zeros((N, N)) for i in range(N_l)]
      conns_back = self.traj.connections_backward[l]
      for i in range(N_l):
        T_i_l = self.traj.features_np[l][i]
        vec[i] += self.Zs[l][i] * outer(T_i_l, T_i_l)
        if i in conns_back:
          for j in conns_back[i]:
            vec[i] += self.w_sstats[l][i] * self.hess_Zs[l-1][j]
          g_vec = np.zeros(N)
          for j in conns_back[i]:
            g_vec += self.grad_Zs[l-1][j]
          vec[i] += self.w_sstats[l][i] * outer(g_vec, T_i_l)
          vec[i] += self.w_sstats[l][i] * outer(T_i_l, g_vec)
      self.hess_Zs.append(vec)
    assert(len(self.hess_Zs) == L)
    self.hess_Z = sum(self.hess_Zs[L-1])
    self.hess_logZ = self.hess_Z / self.Z \
                     - outer(self.grad_Z, self.grad_Z) / (self.Z * self.Z)

class LearningElements2(LearningElementsRef):
  """ Safe implementation for computing the elements, optimized for Numpy.
  
  This implementation is faster and much less readable than LearningElements1,
  so do not try to use it to reimplement an algorithm.
  
  Useful fields:
  - traj: a LearningTrajectory object.
  - theta: the weight vector
  - ...
  """
  # pylint: disable=W0201
  def computeLogZ(self):
    """
    Provides Zs, logZ
    """
    if 'logZ' in self.__dict__:
      return
    L = self.traj.L
    self.logZs = [dot(self.traj.features_np[0], self.theta)]
    assert not np.isnan(self.logZs[0]).any(), \
      (self.traj.features_np[0], self.theta)
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec = MINUS_INF * np.ones((N_l))
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        if i in conns_back:
          l_vec[i] = lse_npy(w[i] + self.logZs[l-1][conns_back[i]])
          assert not np.isnan(l_vec[i]).any()
      assert not np.isnan(l_vec).any()
      self.logZs.append(l_vec)
    assert(len(self.logZs) == L)
    self.logZ = lse_npy(self.logZs[L-1])
    assert not np.isnan(self.logZ).any()
    # Make sure we do not overflow.
    if self.logZ < 100 and self.logZ > -100:
      self.Z = exp(self.logZ)

  def computeGradientLogZ(self):
    if 'grad_logZ' in self.__dict__:
      return
    self.computeLogZ()
    N = len(self.theta)
    L = self.traj.L
    assert not np.isnan(self.theta).any()
    log_grad_Zs0_dirs = self.traj.features_np[0]
    log_grad_Zs0_norms = dot(self.traj.features_np[0], self.theta)
    self.log_grad_Zs_norms = [log_grad_Zs0_norms]
    self.log_grad_Zs_dirs = [log_grad_Zs0_dirs]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec_dirs = np.zeros((N_l, N))
      l_vec_norms = MINUS_INF * np.ones(N_l)
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        vs0_dir = np.array([self.traj.features_np[l][i]])
        vs0_norm = np.array([self.logZs[l][i]])
        if i not in conns_back:
          (n, d) = lse_vec_npy(vs0_dir, vs0_norm)
          l_vec_dirs[i] = d
          l_vec_norms[i] = n
        else:
          vs_dirs = np.vstack((vs0_dir, \
                             self.log_grad_Zs_dirs[l-1][conns_back[i]]))
          vs_norms = np.hstack((vs0_norm, \
                                w[i] + \
                                  self.log_grad_Zs_norms[l-1][conns_back[i]]))
          (n, d) = lse_vec_npy(vs_dirs, vs_norms)
          l_vec_dirs[i] = d
          l_vec_norms[i] = n
      self.log_grad_Zs_norms.append(l_vec_norms)
      self.log_grad_Zs_dirs.append(l_vec_dirs)
    assert(len(self.log_grad_Zs_norms) == L)
    self.log_grad_Z = lse_vec_npy(self.log_grad_Zs_dirs[L-1], 
                                  self.log_grad_Zs_norms[L-1])
    (l_norm, v) = self.log_grad_Z
    if l_norm < 100 and l_norm > -100:
      self.grad_Z = exp(l_norm) * v
    self.grad_logZ = exp(l_norm - self.logZ) * v
    
  def computeHessianLogZ(self):
    """ Hessian of log(Z). """
    if 'hess_logZ' in self.__dict__:
      return
    inf = float('inf')
    self.computeLogZ()
    self.computeGradientLogZ()
    N = len(self.theta)
    L = self.traj.L
    # The initial values:
    N_0 = self.traj.num_choices[0]
    log_hess_Zs0_norms = np.dot(self.traj.features_np[0], self.theta)
    log_hess_Zs0_dirs = np.zeros((N_0, N, N))
    for i in range(N_0):
      log_hess_Zs0_dirs[i] = np.outer(self.traj.features_np[0][i], \
                                      self.traj.features_np[0][i])
    self.log_hess_Zs_norms = [log_hess_Zs0_norms]
    self.log_hess_Zs_dirs = [log_hess_Zs0_dirs]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec_norm = -inf * np.ones(N_l)
      l_vec_dir = np.zeros((N_l, N, N))
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      for i in range(N_l):
        T_i_l = self.traj.features_np[l][i]
        vs0_norm = np.array([self.logZs[l][i]])
        vs0_dir = np.array([outer(T_i_l, T_i_l)])
        if i in conns_back:
          us_norm = self.log_grad_Zs_norms[l-1][conns_back[i]]
          us_dir = self.log_grad_Zs_dirs[l-1][conns_back[i]]
          (l_norm, u_g_vec) = lse_vec_npy(us_dir, us_norm)
          vs_norm = np.hstack((vs0_norm, \
                               w[i] + \
                                 self.log_hess_Zs_norms[l-1][conns_back[i]], \
                               w[i] + l_norm, \
                               w[i] + l_norm))
          M = np.array([outer(u_g_vec, T_i_l)])
          Mt = np.array([outer(T_i_l, u_g_vec)])
          vs_dir = np.vstack((vs0_dir, \
                              self.log_hess_Zs_dirs[l-1][conns_back[i]], \
                              M, Mt))
          (li_vec_norm, li_vec_dir) = lse_vec_npy(vs_dir, vs_norm)
          l_vec_norm[i] = li_vec_norm
          l_vec_dir[i] = li_vec_dir          
        else:
          l_vec_norm[i] = vs0_norm
          l_vec_dir[i] = vs0_dir
      self.log_hess_Zs_dirs.append(l_vec_dir)
      self.log_hess_Zs_norms.append(l_vec_norm)
    assert(len(self.log_hess_Zs_dirs) == L)
    self.log_hess_Z = lse_vec_npy(self.log_hess_Zs_dirs[-1], \
                                  self.log_hess_Zs_norms[-1])
    (l_norm, h) = self.log_hess_Z
    if l_norm < 100 and l_norm > -100:
      self.hess_Z = exp(l_norm) * h
    (l_norm_g, g) = self.log_grad_Z
    self.hess_logZ = np.zeros_like(h)
    if l_norm - self.logZ > -60: 
      self.hess_logZ += exp(l_norm - self.logZ) * h
    if l_norm_g - self.logZ > -30:
      self.hess_logZ -= exp(2 * l_norm_g - 2 * self.logZ) * outer(g, g)


class LearningElements2_(LearningElementsRef):
  """ Safe implementation for computing the elements, optimized for Numpy.
  
  This implementation is faster and much less readable than LearningElements1,
  so do not try to use it to reimplement an algorithm.
  
  Useful fields:
  - traj: a LearningTrajectory object.
  - theta: the weight vector
  - ...
  """
  # pylint: disable=W0201
  def computeLogZ(self):
    """
    Provides Zs, logZ
    """
    if 'logZ' in self.__dict__:
      return
    L = self.traj.L
    self.logZs = [dot(self.traj.features_np[0], self.theta)]
    assert not np.isnan(self.logZs[0]).any(), \
      (self.traj.features_np[0], self.theta)
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec = MINUS_INF * np.ones((N_l))
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        if i in conns_back:
          l_vec[i] = lse_npy(w[i] + self.logZs[l-1][conns_back[i]])
          assert not np.isnan(l_vec[i]).any()
      assert not np.isnan(l_vec).any()
      self.logZs.append(l_vec)
    assert(len(self.logZs) == L)
    self.logZ = lse_npy(self.logZs[L-1])
    assert not np.isnan(self.logZ).any()
    # Make sure we do not overflow.
    if self.logZ < 100 and self.logZ > -100:
      self.Z = exp(self.logZ)

  def computeGradientLogZ(self):
    if 'grad_logZ' in self.__dict__:
      return
    self.computeLogZ()
    N = len(self.theta)
    L = self.traj.L
    assert not np.isnan(self.theta).any()
    log_grad_Zs0_dirs = self.traj.features_np[0]
    log_grad_Zs0_norms = dot(self.traj.features_np[0], self.theta)
    self.log_grad_Zs_norms = [log_grad_Zs0_norms]
    self.log_grad_Zs_dirs = [log_grad_Zs0_dirs]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec_dirs = np.zeros((N_l, N))
      l_vec_norms = MINUS_INF * np.ones(N_l)
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        vs0_dir = [self.traj.features_np[l][i]]
        vs0_norm = [self.logZs[l][i]]
        if i not in conns_back:
          (n, d) = lse_vec_2(vs0_dir, vs0_norm)
          l_vec_dirs[i] = d
          l_vec_norms[i] = n
        else:
          vs_dirs = [vs0_dir, \
                             self.log_grad_Zs_dirs[l-1][conns_back[i]]]
          vs_norms = [vs0_norm, \
                                w[i] + \
                                  self.log_grad_Zs_norms[l-1][conns_back[i]]]
          (n, d) = lse_vec_2(vs_dirs, vs_norms)
          l_vec_dirs[i] = d
          l_vec_norms[i] = n
      self.log_grad_Zs_norms.append(l_vec_norms)
      self.log_grad_Zs_dirs.append(l_vec_dirs)
    assert(len(self.log_grad_Zs_norms) == L)
    self.log_grad_Z = lse_vec_2(self.log_grad_Zs_dirs[L-1], 
                                self.log_grad_Zs_norms[L-1])
    (l_norm, v) = self.log_grad_Z
    if l_norm < 100 and l_norm > -100:
      self.grad_Z = exp(l_norm) * v
    self.grad_logZ = exp(l_norm - self.logZ) * v
    
  def computeHessianLogZ(self):
    """ Hessian of log(Z). """
    if 'hess_logZ' in self.__dict__:
      return
    inf = float('inf')
    self.computeLogZ()
    self.computeGradientLogZ()
    N = len(self.theta)
    L = self.traj.L
    # The initial values:
    N_0 = self.traj.num_choices[0]
    log_hess_Zs0_norms = np.dot(self.traj.features_np[0], self.theta)
    log_hess_Zs0_dirs = np.zeros((N_0, N, N))
    for i in range(N_0):
      log_hess_Zs0_dirs[i] = np.outer(self.traj.features_np[0][i], \
                                      self.traj.features_np[0][i])
    self.log_hess_Zs_norms = [log_hess_Zs0_norms]
    self.log_hess_Zs_dirs = [log_hess_Zs0_dirs]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec_norm = -inf * np.ones(N_l)
      l_vec_dir = np.zeros((N_l, N, N))
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      for i in range(N_l):
        T_i_l = self.traj.features_np[l][i]
        vs0_norm = np.array([self.logZs[l][i]])
        vs0_dir = np.array([outer(T_i_l, T_i_l)])
        if i in conns_back:
          us_norm = self.log_grad_Zs_norms[l-1][conns_back[i]]
          us_dir = self.log_grad_Zs_dirs[l-1][conns_back[i]]
          (l_norm, u_g_vec) = lse_vec_2(us_dir, us_norm)
          vs_norm = (vs0_norm, \
                               w[i] + \
                                 self.log_hess_Zs_norms[l-1][conns_back[i]], \
                               w[i] + l_norm, \
                               w[i] + l_norm)
          M = np.array([outer(u_g_vec, T_i_l)])
          Mt = np.array([outer(T_i_l, u_g_vec)])
          vs_dir = (vs0_dir, \
                              self.log_hess_Zs_dirs[l-1][conns_back[i]], \
                              M, Mt)
          (li_vec_norm, li_vec_dir) = lse_vec_2(vs_dir, vs_norm)
          l_vec_norm[i] = li_vec_norm
          l_vec_dir[i] = li_vec_dir          
        else:
          l_vec_norm[i] = vs0_norm
          l_vec_dir[i] = vs0_dir
      self.log_hess_Zs_dirs.append(l_vec_dir)
      self.log_hess_Zs_norms.append(l_vec_norm)
    assert(len(self.log_hess_Zs_dirs) == L)
    self.log_hess_Z = lse_vec_2(self.log_hess_Zs_dirs[-1], \
                                  self.log_hess_Zs_norms[-1])
    (l_norm, h) = self.log_hess_Z
    if l_norm < 100 and l_norm > -100:
      self.hess_Z = exp(l_norm) * h
    (l_norm_g, g) = self.log_grad_Z
    self.hess_logZ = np.zeros_like(h)
    if l_norm - self.logZ > -60: 
      self.hess_logZ += exp(l_norm - self.logZ) * h
    if l_norm_g - self.logZ > -30:
      self.hess_logZ -= exp(2 * l_norm_g - 2 * self.logZ) * outer(g, g)

class LearningElements1(LearningElementsRef):
  """ Safe implementation for computing the elements.
  
  This implementation performs all computations in log domain. It is slightly
  slower than the reference fast implementation but works for a much larger
  domain of feature values.
  
  Useful fields:
  - traj: a LearningTrajectory object.
  - theta: the weight vector
  - ...
  """
  
  # pylint: disable=W0201
  def computeValue(self):
    """
      Calls LogZ.
      Provides logValue
    """
    if 'logValue' in self.__dict__:
      return
    self.computeSStats()
    self.computeLogV1()
    self.computeLogZ()
    L = self.traj.L
    self.logValues = [dot(self.traj.features_np[0], 
                          self.theta) - dot(self.sstats[0], self.theta)]
    assert not np.isnan(self.logValues[0]).any(), \
      (self.traj.features_np[0], self.theta)
    assert lse(self.logValues[0]) >= 0
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec = MINUS_INF * np.ones((N_l))
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l],
              self.theta) - dot(self.sstats[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        if i in conns_back:
          l_vec[i] = w[i] + lse([self.logValues[l-1][j] \
                                 for j in conns_back[i]])
          assert not np.isnan(l_vec[i]).any()
      self.logValues.append(l_vec)
#      assert(lse(l_vec) >= -1e-2)
      assert not np.isnan(l_vec).any()
    assert(len(self.logValues) == L)
    self.logValue = -lse(self.logValues[L-1])
    assert not np.isnan(self.logZ).any()
#    assert self.logV1 <= self.logZ, (self.logV1, self.logZ, self.sstats)


  def computeLogV1(self):
    """ Log V1.
    """
    if 'logV1' in self.__dict__:
      return
    self.computeSStats()
    L = self.traj.L
    # Expected sufficient statistics:
    self.logV1 = sum([dot(self.sstats[l], self.theta) for l in range(L)])
  
  # pylint: disable=W0201
  def computeLogZ(self):
    """
    Provides Zs, logZ
    """
    if 'logZ' in self.__dict__:
      return
    self.computeLogV1()
    L = self.traj.L
    self.logZs = [dot(self.traj.features_np[0], self.theta)]
    assert not np.isnan(self.logZs[0]).any(), \
      (self.traj.features_np[0], self.theta)
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec = MINUS_INF * np.ones((N_l))
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        if i in conns_back:
          l_vec[i] = w[i] + lse([self.logZs[l-1][j] for j in conns_back[i]])
          assert not np.isnan(l_vec[i]).any()
      assert not np.isnan(l_vec).any()
      self.logZs.append(l_vec)
    assert(len(self.logZs) == L)
    self.logZ = lse(self.logZs[L-1])
    assert not np.isnan(self.logZ).any()
    # Make sure we do not overflow.
    if self.logZ < 100 and self.logZ > -100:
      self.Z = exp(self.logZ)

  def computeGradientLogZ(self):
    if 'grad_logZ' in self.__dict__:
      return
    self.computeLogZ()
    inf = float('inf')
    N = len(self.theta)
    L = self.traj.L
    # The initial values:
    N_0 = self.traj.num_choices[0]
    assert not np.isnan(self.theta).any()
    log_grad_Zs0 = [(dot(self.traj.features_np[0][i], self.theta), \
                     self.traj.features_np[0][i]) for i in range(N_0)]
    self.log_grad_Zs = [log_grad_Zs0]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec = [(-inf, np.zeros(N)) for i in range(N_l)]
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      assert not np.isnan(w).any()
      for i in range(N_l):
        vs = [(self.logZs[l][i], self.traj.features_np[l][i])]
        assert not np.isnan(vs[0][0]).any()
        assert not np.isnan(vs[0][1]).any()
        if i in conns_back:
          for j in conns_back[i]:
            (l_norm, v) = self.log_grad_Zs[l-1][j]
            assert not np.isnan(v).any()
            assert not np.isnan(l_norm).any()
            vs.append((w[i] + l_norm, v))
        l_vec[i] = lse_vec(vs)
        assert not np.isnan(l_vec[i][0]).any(), (l_vec[i], vs)
        assert not np.isnan(l_vec[i][1]).any(), (l_vec[i], vs)
      self.log_grad_Zs.append(l_vec)
    assert(len(self.log_grad_Zs) == L)
    self.log_grad_Z = lse_vec(self.log_grad_Zs[L-1])
    (l_norm, v) = self.log_grad_Z
    if l_norm < 100 and l_norm > -100:
      self.grad_Z = exp(l_norm) * v
    self.grad_logZ = exp(l_norm - self.logZ) * v
    
  def computeHessianLogZ(self):
    """ Hessian of log(Z). """
    if 'hess_logZ' in self.__dict__:
      return
    inf = float('inf')
    self.computeLogZ()
    self.computeGradientLogZ()
    N = len(self.theta)
    L = self.traj.L
    # The initial values:
    N_0 = self.traj.num_choices[0]
    log_hess_Zs0 = []
    for i in range(N_0):
      T_i_0 = self.traj.features_np[0][i]
      log_hess_Zs0.append((dot(T_i_0, self.theta), outer(T_i_0, T_i_0)))
    self.log_hess_Zs = [log_hess_Zs0]
    # Recursion:
    for l in range(1, L):
      N_l = self.traj.num_choices[l]
      l_vec = [(-inf, np.zeros((N, N))) for i in range(N_l)]
      conns_back = self.traj.connections_backward[l]
      w = dot(self.traj.features_np[l], self.theta)
      for i in range(N_l):
        T_i_l = self.traj.features_np[l][i]
        vs = [(self.logZs[l][i], outer(T_i_l, T_i_l))]
        if i in conns_back:
          assert conns_back[i], (i, conns_back)
          for j in conns_back[i]:
            (l_norm, h) = self.log_hess_Zs[l-1][j]
            vs.append((w[i] + l_norm, h))
          log_g_vec = lse_vec([self.log_grad_Zs[l-1][j] for j in conns_back[i]])
          (l_norm, u_g_vec) = log_g_vec
          vs.append((w[i] + l_norm, outer(u_g_vec, T_i_l)))
          vs.append((w[i] + l_norm, outer(T_i_l, u_g_vec)))
        l_vec[i] = lse_vec(vs)
      self.log_hess_Zs.append(l_vec)
    assert(len(self.log_hess_Zs) == L)
    self.log_hess_Z = lse_vec(self.log_hess_Zs[-1])
    (l_norm, h) = self.log_hess_Z
    if l_norm < 100 and l_norm > -100:
      self.hess_Z = exp(l_norm) * h
    (l_norm_g, g) = self.log_grad_Z
    self.hess_logZ = exp(l_norm - self.logZ) * h\
                     - exp(2 * l_norm_g - 2 * self.logZ) * outer(g, g)
