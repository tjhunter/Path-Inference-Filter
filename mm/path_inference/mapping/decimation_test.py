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
Created on Nov 10, 2011

@author: tjhunter
'''

from mm.path_inference.structures import Path, State
from decimation import merge_path, decimate_path_simple, merge_path_sequence

def test_merge_path_simple():
  """ test_merge_path_simple
  """
  s1 = State(1, 1.0, None)
  s2 = State(2, 3.0, None)
  s3 = State(3, 1.0, None)
  p1 = Path(s1, [1, 2], s2)
  p2 = Path(s2, [2, 3], s3)
  p = merge_path(p1, p2)
  assert p.start == s1
  assert p.end == s3
  assert p.links == [1, 2, 3]

def test_simple_decimation_1():
  """ test_simple_decimation_1
  """
  s1_0 = State(1, 0.0, None)
  s1_1 = State(2, 0.0, None)
  s1_2 = State(3, 0.0, None)
  s2_0 = State(1, 1.0, None)
  s2_1 = State(2, 2.0, None)
  s2_2 = State(3, 3.0, None)
  paths = [Path(s1_0, [1], s2_0), Path(s1_1, [2], s2_1), Path(s1_2, [3], s2_2)]
  trans1 = [(0, 0), (1, 1), (2, 2)]
  trans2 = [(0, 0), (1, 1), (2, 2)]
  start_mapping = {1:0}
  end_mapping = {1:0}
  (new_trans1, new_paths, new_trans2, paths_mapping) = \
    decimate_path_simple(start_mapping, trans1, paths, trans2, end_mapping)
  assert paths_mapping == {1:0}, paths_mapping
  assert new_trans1 == [(0, 0)], new_trans1
  assert new_trans2 == [(0, 0)], new_trans2
  assert len(new_paths) == 1
  assert new_paths[0].links == [2]

def test_simple_decimation_2():
  """ test_simple_decimation_2
  """
  s1_0 = State(0, 0.0)
  #s1_1 = State(1, 0.0)
  s1_2 = State(2, 0.0)
  s2_0 = State(0, 1.0)
  s2_1 = State(1, 1.0)
  s2_2 = State(2, 1.0)
  paths = [Path(s1_0, [0, 1], s2_1), \
           Path(s1_2, [2, 1], s2_1), \
           Path(s1_0, [0, 0], s2_0), \
           Path(s1_2, [2, 2], s2_2)]
  print paths
  trans1 = [(0, 0), (2, 1), (0, 2), (2, 3)]
  trans2 = [(0, 1), (1, 1), (2, 0), (3, 2)]
  start_mapping = {2:0}
  end_mapping = {1:0, 2:1}
  (new_trans1, new_paths, new_trans2, paths_mapping) = \
    decimate_path_simple(start_mapping, trans1, paths, trans2, end_mapping)
  assert paths_mapping == {1:0, 3:1}, paths_mapping
  assert len(new_paths) == 2
  assert new_paths[0].links == [2, 1]
  assert new_paths[1].links == [2, 2]
  assert new_trans1 == [(0, 0), (0, 1)], new_trans1
  assert new_trans2 == [(0, 0), (1, 1)], new_trans2

def test_merge_path_sequence_1():
  """ test_merge_path_sequence_1
  """
  sa_0 = State(0, 0.0, None)
  sa_1 = State(1, 0.0, None)
  sa_2 = State(2, 0.0, None)
  sb_0 = State(0, 1.0, None)
  sb_1 = State(1, 2.0, None)
  sb_2 = State(2, 3.0, None)
  sc_0 = State(0, 0.0, None)
  sc_1 = State(1, 0.0, None)
  sc_2 = State(2, 0.0, None)
  paths_a = [Path(sa_0, [0], sb_0), \
             Path(sa_1, [1], sb_1), \
             Path(sa_2, [2], sb_2)]
  paths_b = [Path(sb_0, [0], sc_0), \
             Path(sb_1, [1], sc_1), \
             Path(sb_2, [2], sc_2)]
  trans1_a = [(0, 0), (1, 1), (2, 2)]
  trans1_b = [(0, 0), (1, 1), (2, 2)]
  trans2_a = [(0, 0), (1, 1), (2, 2)]
  trans2_b = [(0, 0), (1, 1), (2, 2)]
  best_idx_a = 1
  best_idx_b = 1
  (new_trans1, new_paths, new_trans2, new_best_idx) = \
    merge_path_sequence(trans1_a, paths_a, trans2_a, trans1_b, paths_b, \
                        trans2_b, best_idx_a, best_idx_b)
  assert len(new_paths) == 3
  assert new_trans1 == [(0, 0), (1, 1), (2, 2)]
  assert new_trans2 == [(0, 0), (1, 1), (2, 2)]
  assert new_best_idx == 1

def test_merge_path_sequence_2():
  """ test_merge_path_sequence_2
  """
  sa_0 = State(0, 0.0)
  sa_1 = State(1, 0.0)
  sb_0 = State(0, 1.0)
  sb_1 = State(1, 1.0)
  sc_0 = State(0, 2.0)
  sc_1 = State(1, 2.0)
  paths_a = [Path(sa_0, [0], sb_0), \
             Path(sa_0, [0, 1], sb_1), \
             Path(sa_1, [1, 0], sb_0), \
             Path(sa_1, [1], sb_1)]
  paths_b = [Path(sb_0, [0], sc_0), \
             Path(sb_0, [0, 1], sc_1), \
             Path(sb_1, [1, 0], sc_0), \
             Path(sb_1, [1], sc_1)]
  trans1_a = [(0, 0), (0, 1), (1, 2), (1, 3)]
  trans2_a = [(0, 0), (1, 1), (2, 0), (3, 1)]
  trans1_b = [(0, 0), (0, 1), (1, 2), (1, 3)]
  trans2_b = [(0, 0), (1, 1), (2, 0), (3, 1)]
  best_idx_a = 0
  best_idx_b = 1
  (new_trans1, new_paths, new_trans2, new_best_idx) = \
    merge_path_sequence(trans1_a, paths_a, trans2_a, trans1_b, paths_b, \
                        trans2_b, best_idx_a, best_idx_b)
  assert len(new_paths) == 8, len(new_paths)
  assert new_trans1 == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7)], new_trans1
  assert new_trans2 == [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (6, 0), (7, 1)], new_trans2
  assert new_best_idx == 1, new_best_idx
