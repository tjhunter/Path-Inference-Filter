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
import collections

class FlowAnalysis(object):
  """ Flow analysis: finds all reachable points from a start point.
  """
  def __init__(self):
    self.count = 0
    self.flow_units = []
    self.reachable = []
    self.num_reachable = []
    self.previous_sc = None # Debugging
  def call(self, dct):
    """ Calling function, to create a closure.
    """
    (trans1, paths_dct, trans2, sc_dct) = dct
    sc = decode_StateCollection(sc_dct)
    num_paths = len(paths_dct)
    num_states = len(sc.states)
    paths = [decode_Path(path_dct) for path_dct in paths_dct]
    # Due to a bug in the way the closures are used, we nned
    # to make sure we are not in the first element.
    # Make a dictionary of previous->paths
    dic1 = collections.defaultdict(list)
    for (u, v) in trans1:
      # Debugging: make sure the path corresponds to the previous point
      assert u >= 0
      assert v >= 0
      assert v < num_paths
      if self.previous_sc:
        assert u < len(self.previous_sc.states)
        # Make sure it is proper connectivity
        state = self.previous_sc.states[u]
        path = paths[v]
        assert state.link_id == path.links[0]
        assert state.link_id == path.start.link_id
        assert state.offset == path.start.offset
      dic1[u].append(v)
    dic2 = collections.defaultdict(list)
    for (v, u) in trans2:
      assert u >= 0
      assert v >= 0
      assert v < num_paths
      assert u < num_states
      # Check connectivity
      state = sc.states[u]
      path = paths[v]
      if state.link_id != path.links[-1]:
        print sc.states
        print path
      assert state.link_id == path.links[-1], \
        (path.links[-1], state.link_id, v, u, state, path)
      assert state.link_id == path.end.link_id, (state.link_id, \
                                                 path.end.link_id)
      assert state.offset == path.end.offset, (state.offset, path.end.offset)
      dic2[v].append(u)
    reachable_paths = set(reduce(lambda l1, l2:l1+l2, \
                                 [dic1[u] for u in self.reachable], []))
    if not reachable_paths:
      print("No reachable paths")
    reachable_states = set(reduce(lambda l1, l2:l1+l2, \
                                  [dic2[u] for u in reachable_paths], []))
    self.num_reachable.append(len(reachable_states))
    if not reachable_states:
      print 'break at %i' % self.count
      self.reachable = range(num_states)
      self.flow_units.append([])
    else:
      self.reachable = reachable_states
    self.flow_units[-1].append(self.count)
    self.count += 1
    self.previous_sc = sc
    return None

