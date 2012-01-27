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
Created on Nov 13, 2011

@author: tjhunter
'''
from mm.path_inference.json import encode_Path, decode_StateCollection, \
  decode_Path, encode_StateCollection
from decimation import decimate_point, decimate_path_simple, merge_path_sequence

class HFDecimation(object):
  """ Builds the reference track.
  
  Sends back the new elements of the trajectory to the caller function.
  """
  def __init__(self, decimation_factor, probas, viterbi_idxs):
    self.count = 0
    # Indexes of the most likely element at each step.
    # Invariant: the start point is never null, except at start.
    # Invariant: if the end point is not null, the paths, start_trans
    # and end_trans are well defined and between the start and end point.
    self.most_likely_indexes = []
    self.previous_correspondance = None
    self.decimation_factor = decimation_factor
    self.viterbi_idxs = viterbi_idxs
    self.probas = probas
    # SC in decimated format
    # Start point of the sequence
    self.start_point = None
    # index mapping of the start of the sequence.
    self.start_mapping = None
    # SC in decimate format
    # End point of the sequence
    self.end_point = None
    # Mapping of the end poit of the sequence.
    self.end_mapping = None
    # Start transition of the current path.
    self.start_trans = None
    # End transition of the current path.
    self.end_trans = None
    # Current paths in the transition between start point and end point.
    self.paths = None
    # Current best index amongst the paths.
    self.best_idx = None
  
  def call(self, dct):
    """ Closure.
    """
    # All the hard elements are already coded in mapping.
    # Here is how the mapping procedure works in high frequency:
    # At first, a new point is added to the beginning, with empty paths.
    # When a new point (with its paths) arrives, two cases:
    # We were right after the start point, then we simply add
    # the path and the point. Otherwise, we decimate paths and points,
    # and we merge the paths together.
    (trans1, paths_dct, trans2, sc_dct) = dct
    paths = [decode_Path(path_dct) for path_dct in paths_dct]
    del paths_dct
    sc = decode_StateCollection(sc_dct)
    del sc_dct
    # The index of sc
    point_idx = 2 * self.count
    # The index of the paths
    path_idx = 2 * self.count - 1
    self.count += 1
    (new_decim_sc, new_end_mapping) = \
      decimate_point(sc, self.probas[point_idx], self.viterbi_idxs[point_idx])
    new_most_likely_sc_idx = None
    if point_idx >= 0:
      assert self.viterbi_idxs[point_idx] in new_end_mapping, \
        (point_idx, new_end_mapping)
      new_most_likely_sc_idx = new_end_mapping[self.viterbi_idxs[point_idx]]
    # First element??
    if not self.start_point:
      self.start_point = new_decim_sc
      self.start_mapping = new_end_mapping
      assert new_most_likely_sc_idx is not None
      self.most_likely_indexes.append(new_most_likely_sc_idx)
      return ([], [], [], encode_StateCollection(self.start_point))
    # Try to add a new elment:
    if self.paths is None:
      assert self.start_mapping is not None
      assert self.start_point is not None
      # Start a new set of paths
      (new_trans1, decimated_paths, new_trans2, paths_mapping) = \
        decimate_path_simple(self.start_mapping, trans1, paths, \
                             trans2, new_end_mapping)
      assert self.viterbi_idxs[point_idx] in new_end_mapping
      assert self.viterbi_idxs[path_idx] in paths_mapping
      assert (self.start_mapping[self.viterbi_idxs[path_idx-1]], \
                paths_mapping[self.viterbi_idxs[path_idx]]) in new_trans1
      assert (paths_mapping[self.viterbi_idxs[path_idx]], \
                new_end_mapping[self.viterbi_idxs[point_idx]]) in new_trans2
      self.end_point = new_decim_sc
      self.end_mapping = new_end_mapping
      self.start_trans = new_trans1
      self.paths = decimated_paths
      assert self.paths, self.paths
      self.best_idx = paths_mapping[self.viterbi_idxs[path_idx]]
      self.end_trans = new_trans2
    else:
      assert self.start_mapping is not None
      assert self.start_point is not None
      assert self.start_trans is not None
      assert self.end_trans is not None
      assert self.paths is not None
      # First decimate the paths
      (new_trans1, decimated_paths, new_trans2, paths_mapping) = \
        decimate_path_simple(self.end_mapping, trans1, paths, trans2, \
                             new_end_mapping)
      assert self.viterbi_idxs[path_idx] in paths_mapping
      assert self.viterbi_idxs[path_idx-1] in self.end_mapping
      assert (self.end_mapping[self.viterbi_idxs[path_idx-1]], \
              paths_mapping[self.viterbi_idxs[path_idx]]) in new_trans1
      assert (paths_mapping[self.viterbi_idxs[path_idx]], \
              new_end_mapping[self.viterbi_idxs[point_idx]]) in new_trans2
      best_idx2 = paths_mapping[self.viterbi_idxs[path_idx]]
      # Merge the paths together
      (merged_trans1, merged_paths, merged_trans2, merged_best_idx) = \
        merge_path_sequence(self.start_trans, self.paths, self.end_trans, \
                            new_trans1, decimated_paths, new_trans2, \
                            self.best_idx, best_idx2)
      self.end_point = new_decim_sc
      self.end_mapping = new_end_mapping
      self.start_trans = merged_trans1
      self.paths = merged_paths
      self.best_idx = merged_best_idx
      self.end_trans = merged_trans2
    # Time to send a new element to the output and restart?  
    if (self.count-1) % self.decimation_factor == 0:
      assert self.paths
      assert self.end_trans
      assert self.start_trans
      assert self.best_idx is not None
      encoded_paths = [encode_Path(path) for path in self.paths]
      print len(encoded_paths), " paths", len(self.end_point.states), " states"
      result = (self.start_trans, encoded_paths, \
                self.end_trans, encode_StateCollection(self.end_point))
      # Adding the most likely index of the path and of the next point.
      self.most_likely_indexes.append(self.best_idx)
      assert new_most_likely_sc_idx is not None
      self.most_likely_indexes.append(new_most_likely_sc_idx)
      # Restart computations:
      self.start_point = self.end_point
      self.start_mapping = self.end_mapping
      del self.paths
      self.start_trans = None
      self.end_trans = None
      self.paths = None
      self.best_idx = None
      return result
    # Nothing to return for this input, continuing.
    return None
