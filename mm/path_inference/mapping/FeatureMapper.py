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
'''
Created on Nov 8, 2011

@author: tjhunter
'''

from mm.path_inference.json import decode_Path, decode_StateCollection
from utils import mm_dis
import numpy as np

def path_length(p):
  """ Length of a path. """
  res = 0.0
  for (latlng1, latlng2) in zip(p.latlngs[1:], p.latlngs[:-1]):
    res += mm_dis(latlng1, latlng2)
  return res


class FeatureMapper(object):
  ''' Maps a track to feature vectors interleaved with connection data.
  
  Each vector contains first all the information relative to a path, and
  then to a state.
  For each path and each state the feature vectors are stored together
  row-wise in a 2d array.
  '''
  NUM_PATH_FEATURES = 1
  NUM_STATE_FEATURES = 1

  def __init__(self):
    self.count = 0
    self.feature_trajs = []
    self.num_path_features = FeatureMapper.NUM_PATH_FEATURES
    self.num_state_features = FeatureMapper.NUM_STATE_FEATURES
  
  def path_features(self, path):
    """ Feature subvector for paths.
    """
    return [-path_length(path)]
  
  def state_features(self, sc, i):
    """ Feature subvector for states.
    """
    state = sc.states[i]
    d = mm_dis(state.gps_pos, sc.gps_pos)
    return [-0.5 * d * d]
  
  def call(self, dct):
    """ Closure.
    """
    (trans1, paths_dct, trans2, sc_dct) = dct
    num_paths = len(paths_dct)
    vector_size = self.num_path_features+self.num_state_features
    empty_paths = [0 for _ in range(self.num_path_features)]
    empty_states = [0 for _ in range(self.num_state_features)]
    feats_paths = np.zeros((num_paths, vector_size))
    for i in range(num_paths):
      path_dct = paths_dct[i]
      path = decode_Path(path_dct)
      phi = np.array(self.path_features(path) + empty_states)
      feats_paths[i, ::] = phi
    sc = decode_StateCollection(sc_dct)
    num_states = len(sc.states)
    feats_states = np.zeros((num_states, vector_size))
    for j in range(len(sc.states)):
      phi = np.array(empty_paths + self.state_features(sc, j))
      feats_states[j, ::] = phi
    if self.count == 0:
      self.feature_trajs.append(feats_states)
    else:
      # Debugging checks
      num_previous_states = len(self.feature_trajs[-1])
      for (u, v) in trans1:
        assert u >= 0
        assert v >= 0
        assert u < num_previous_states, ('trans1', (u, v), u)
        assert v < num_paths, ('trans1', (u, v), v)
      for (u, v) in trans2:
        assert u >= 0
        assert v >= 0
        assert u < num_paths, ('trans2', (u, v), u)
        assert v < num_states, ('trans2', (u, v), v)
      self.feature_trajs.append(trans1)
      self.feature_trajs.append(feats_paths)
      self.feature_trajs.append(trans2)
      self.feature_trajs.append(feats_states)
    self.count += 1
    return None
