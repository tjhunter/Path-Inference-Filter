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
from mm.path_inference.structures import Path, StateCollection
from collections import defaultdict

def decimate_point(sc, proba_vec, best_idx=None):
  """ Returns a pair that contains: a new state collection with the decimated
  points and a list of indexes that provides a correspondence between the old
  points and the new points.
  
  Parameters:
  - best_idx: if specified, the returned state collection will contain the 
  spot specified by the corresponding index.
  
  Returns: (new_sc, mapping) with new_sc a new StateCollection and mapping
  a mapping from old indexes (in sc) to new indexes (in new_sc)
  """
  n = len(sc.states)
  assert n == len(proba_vec)
  # Build a dictionary that maps a link id to the most likely point on that
  # link id.
  d = {}
  for i in range(n):
    state = sc.states[i]
    link_id = state.link_id
    if link_id not in d:
      d[link_id] = i
    else:
      if proba_vec[d[link_id]] < proba_vec[i]:
        d[link_id] = i
  # Make sure we store the best index if needed be:
  if best_idx is not None:
    d[sc.states[best_idx].link_id] = best_idx
  new_states = []
  mapping = {}
  j = 0
  for i in d.values():
    new_states.append(sc.states[i])
    mapping[i] = j
    j += 1
  new_sc = StateCollection(sc.id, new_states, sc.gps_pos, sc.time)
  return (new_sc, mapping)

def merge_path(path1, path2):
  """ Merges two path objects together.
  
  Assumes the paths are linked together by the same state.
  """
  assert path1.end == path2.start, (path1.end, path2.start, path1, path2)
  links = path1.links[:-1] + path2.links
  latlngs = None
  if path1.latlngs and path2.latlngs:
    latlngs = path1.latlngs[:-1] + path2.latlngs
  return Path(path1.start, links, path2.end, latlngs)

def decimate_path_simple(start_mapping, trans1, paths, trans2, end_mapping):
  """ 
  Returns (trans1, paths, trans2, paths_mapping) in new mapping
  Args:
  - start_mapping: dictionary(old index -> new index), mapping of the start point
  - end_mapping: dictionary(old index -> new index), mapping for the end point
  - trans1: list of (old point index, old path index), correspondance index
  - paths: list of paths
  - trans2: list of (old path index, old point index), correspondance index
  """
  trans1_by_paths = dict([(path_idx, pt_idx) for (pt_idx, path_idx) in trans1])
  trans2_by_paths = dict(trans2)
  new_trans1 = []
  new_trans2 = []
  new_paths = []
  paths_mapping = {}
  for path_idx in range(len(paths)):
    if trans1_by_paths[path_idx] in start_mapping \
       and trans2_by_paths[path_idx] in end_mapping:
      p = paths[path_idx]
      new_path_idx = len(new_paths)
      new_paths.append(p)
      paths_mapping[path_idx] = new_path_idx
      new_start_pt_idx = start_mapping[trans1_by_paths[path_idx]]
      new_trans1.append((new_start_pt_idx, new_path_idx))
      new_trans2.append((new_path_idx, end_mapping[trans2_by_paths[path_idx]]))
  return (new_trans1, new_paths, new_trans2, paths_mapping)  

def merge_path_sequence(trans1_a, paths_a, trans2_a, trans1_b, paths_b, \
                        trans2_b, best_idx_a, best_idx_b):
  """ Merges two paths sequences and preserves the transition information as
  well as some specific best index about the sequence.
  
  Args:
  - trans1_a: transitions from point a to path a->b
  - paths_a: collection of paths a->b
  - trans2_a: transitions from paths a->b to point b
  - trans1_b: transitions from point b to paths b->c
  - paths_b: collection of paths b->c
  - trans2_b: transition from paths b->c to point c
  - best_idx_a, best_idx_b: best indexes.  
  Returns (trans1, paths, trans2, best_idx)
  """
  # Aggregate the second paths by their start point:
  trans1_b_by_points = defaultdict(list)
  for (pt_b_idx, pa_b_idx) in trans1_b:
    trans1_b_by_points[pt_b_idx].append(pa_b_idx)
  trans2_a_by_paths = dict(trans2_a)
  trans2_b_by_paths = dict(trans2_b)
  new_trans1 = []
  new_trans2 = []
  new_paths = []
  best_idx = None
  for (pt_a_idx, pa_a_idx) in trans1_a:
    path_a = paths_a[pa_a_idx]
    pt_b_idx = trans2_a_by_paths[pa_a_idx]
    if pt_b_idx in trans1_b_by_points:
      for pa_b_idx in trans1_b_by_points[pt_b_idx]:
        path_b = paths_b[pa_b_idx]
        pt_end_idx = trans2_b_by_paths[pa_b_idx]
        new_path = merge_path(path_a, path_b)
        new_path_idx = len(new_paths)
        new_trans1.append((pt_a_idx, new_path_idx))
        new_trans2.append((new_path_idx, pt_end_idx))
        new_paths.append(new_path)
        if pa_a_idx == best_idx_a and pa_b_idx == best_idx_b:
          best_idx = new_path_idx
  assert best_idx is not None
  return (new_trans1, new_paths, new_trans2, best_idx)
