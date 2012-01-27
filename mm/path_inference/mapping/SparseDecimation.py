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
Created on Nov 27, 2011

@author: tjhunter
'''
from mm.path_inference.json import encode_Path, decode_StateCollection, \
  decode_Path, encode_StateCollection
from decimation import merge_path

class SparseDecimation(object):
  """ Builds the reference track for some decimation rates higher than one 
  second. Drops some paths if too many paths are computed between two points,
  but ensures that the true path is kept.
  
  Sends back the new elements of the trajectory to the caller function as JSON
  dictionaries.

  Takes a one-second decimated track and builds from it a new track with a higher
  decimation rate.
  
  This class is somewhat simpler than HFDecimation as it does not need to merge
  back some paths together, and it does not need to perform point decimation.
  """
  def __init__(self, decimation_factor, viterbi_idxs, path_builder):
    self.count = 0
    self.path_builder = path_builder
    # Indexes of the most likely element at each step.
    # Invariant: the start point is never null, except at start.
    # Invariant: if the end point is not null, the paths, start_trans
    # and end_trans are well defined and between the start and end point.
    self.most_likely_indexes = []
    self.previous_correspondance = None
    self.decimation_factor = decimation_factor
    self.viterbi_idxs = viterbi_idxs
    # SC in decimated format
    # Start point of the sequence
    self.start_point = None
    # The best path from the start point to the end point.
    # It is unique.
    self.best_path = None

  def call(self, dct):
    """ Closure.
    """
    (_, paths_dct, _, sc_dct) = dct
    sc = decode_StateCollection(sc_dct)
    del sc_dct
    # The index of sc
    point_idx = 2 * self.count
    # The index of the paths
    path_idx = 2 * self.count - 1
    self.count += 1
    new_most_likely_sc_idx = self.viterbi_idxs[point_idx]
    # First element??
    if not self.start_point:
      self.start_point = sc
      assert new_most_likely_sc_idx is not None
      self.most_likely_indexes.append(new_most_likely_sc_idx)
      return ([], [], [], encode_StateCollection(self.start_point))
    # Only decode the most likely path, we do not need the other paths.
    new_best_path = decode_Path(paths_dct[self.viterbi_idxs[path_idx]])
    del paths_dct
    # Try to add a new element:
    # All this code is much more complicated than it should be now.
    if self.best_path is None:
      assert self.start_point is not None
      self.best_path = new_best_path
    else:
      assert self.start_point is not None
      self.best_path = merge_path(self.best_path, new_best_path)
      assert self.best_path.start in self.start_point.states
      assert self.best_path.end in sc.states
    # Time to send a new element to the output and restart?  
    if (self.count-1) % self.decimation_factor == 0:
      # Time to find all the other paths
      (other_trans1, other_paths, other_trans2) = \
        self.path_builder.getPathsBetweenCollections(self.start_point, sc)
      # If we have the first path already in, no need to integrate it:
      try:
        best_path_idx = other_paths.index(self.best_path)
        new_trans1 = other_trans1
        new_paths = other_paths
        new_trans2 = other_trans2
      except ValueError:
        # We need to append it:
        best_path_idx = len(other_paths)
        prev_best_idx = self.most_likely_indexes[-1]
        new_trans1 = other_trans1 + [(prev_best_idx, best_path_idx)]
        new_paths = other_paths + [self.best_path]
        new_trans2 = other_trans2 + [(best_path_idx, new_most_likely_sc_idx)]
      
      encoded_paths = [encode_Path(path) for path in new_paths]
      print len(encoded_paths), " paths", len(sc.states), " states",
      if len(other_paths) != len(new_paths):
        print '(forced insertion)'
      else:
        print ''
      result = (new_trans1, encoded_paths, \
                new_trans2, encode_StateCollection(sc))
      # Adding the most likely index of the path and of the next point.
      self.most_likely_indexes.append(best_path_idx)
      assert new_most_likely_sc_idx is not None
      self.most_likely_indexes.append(new_most_likely_sc_idx)
      # Restart computations:
      self.start_point = sc
      self.best_path = None
      return result
    # Nothing to return for this input, continuing.
    return None
