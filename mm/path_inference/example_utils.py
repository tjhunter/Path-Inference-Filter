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
Created on Jan 26, 2012

@author: tjhunter
'''
# pylint: disable=W0105
import random
random.seed = 2 # Fixing the seed to reproduce results.
from mm.path_inference.structures import State, LatLng, StateCollection
from mm.path_inference.path_builder import PathBuilder

""" We sample some observation links from the trajectory.
"""
def get_pos(link_id, off, lattice_link_length):
  """ Returns the (lat, lng) that corrsponds to this position in the network.
  
  Paramters:
   - link_id: a tupe of (start node, end node)
   - off: offset length (meters)
  """
  ((x0, y0), (x1, y1)) = link_id
  u = off / lattice_link_length
  lat =  lattice_link_length * (u * x1 + (1-u) * x0)
  lng =  lattice_link_length * (u * y1 + (1-u) * y0)
  return (lat, lng)


def create_trajectory_for_lattice_network(lattice_size, lattice_link_length,
                                          sigma, obs_sigma, 
                                          num_observations, means):

  """ Simulate a vehicle travelling on the maze.
  
  This is implemented as a random walk going from one node to the other
  in the lattice.
  
  Parameters:
   - lattice_size
   - lattice_link_length
   - sigma
   - obs_sigma
   - num_observations
   - means: dictionary of mean travel times on each link: link_id -> mean_tt 
            (in seconds)
  """
  traj_nodes = [(0, 0)]
  while True:
    (x, y) = traj_nodes[-1]
    if x == lattice_size-1 and y == lattice_size-1:
      break
    if x == lattice_size-1:
      traj_nodes.append((x, y+1))
    elif y == lattice_size-1:
      traj_nodes.append((x+1, y))
    else:
      if random.randint(0, 1) == 0:
        traj_nodes.append((x+1, y))
      else:
        traj_nodes.append((x, y+1))
  
  num_traj_nodes = len(traj_nodes)
  
  """ Some random times on each link:
  """
  travel_times = {}
  for x in range(lattice_size):
    for y in range(lattice_size):
      n = (x, y)
      n_right = (x+1, y)
      n_up = (x, y+1)
      travel_times[(n, n_right)] = random.gauss(means[(n, n_right)], sigma)
      travel_times[(n, n_up)] = random.gauss(means[(n, n_up)], sigma)
      
  observation_link_indexes = [0] + random.sample(range(1, num_traj_nodes-2), 
                                        num_observations) + [num_traj_nodes-2] 
  """ Observation contains a list of states: this will be our observations. """
  observations = []
  for n_idx in observation_link_indexes:
    # The link index
    link = (traj_nodes[n_idx], traj_nodes[n_idx+1])
    # Pick a random offset
    offset = lattice_link_length * random.random()
    # The simulated GPS observation
    # Lat and lng in meters
    (true_lat, true_lng) = get_pos(link, offset, lattice_link_length)
    pos = LatLng(lat=random.gauss(true_lat, obs_sigma),
                 lng=random.gauss(true_lng, obs_sigma))
    # Do a few projections in the neighboring links.
    states = []
    ((x0, y0), _) = link
    for x in range(x0-2, x0+2):
      for y in range(y0-2, y0+2):
        if x < 0 or y < 0:
          continue
        for (dx, dy) in [(0, 1), (1, 0)]:
          x_end = x+dx
          y_end = y+dy      
          if x_end >= lattice_size or y_end >= lattice_size:
            continue
          proj_link = ((x, y), (x_end, y_end))
          # Find the projection on the link
          (lat_start, lng_start) = get_pos(proj_link, 0, lattice_link_length)
          off = dx * (pos.lat - lat_start) + dy * (pos.lng - lng_start)
          off = max(min(off, lattice_link_length), 0)
          s = State(proj_link, off, 
                    pos=LatLng(*get_pos(proj_link, off, 
                                        lattice_link_length))) #Using magic
          states.append(s)
    sc = StateCollection(None, states, pos, None)
    sc.true_state = (link, offset)
    observations.append(sc)
  
  """ Simulate some travel times.
  """
  observed_travel_times = []
  traj_links = zip(traj_nodes[:-1], traj_nodes[1:])
  for i in range(num_observations+1):
    tt = 0
    (start_link, start_offset) = observations[i].true_state
    tt += travel_times[start_link] * (1 - start_offset / lattice_link_length)
    links_between = traj_links[observation_link_indexes[i]+1:\
                               observation_link_indexes[i+1]]
    for l in links_between:
      tt += travel_times[l]
    (end_link, end_offset) = observations[i+1].true_state
    tt += travel_times[end_link] * end_offset / lattice_link_length
    observed_travel_times.append(tt)

  return (observations, observed_travel_times)
  

class LatticePathBuilder(PathBuilder):
  """ Path builder for a lattice network.
  
  The path builder will generate all the paths between two state collections
  mapped in the lattice network.
  """
  
  def get_paths_lattice(self, start_node, end_node):
    """ Computes all the paths between a start node and an end node.
    """
    (x0, y0) = start_node
    (x1, y1) = end_node
    if start_node == end_node:
      return [[]]
    if x0 > x1 or y0 > y1:
      return []
    next_up = (x0, y0+1)
    next_right = (x0+1, y0)
    res = []
    for p  in self.get_paths_lattice(next_up, end_node):
      res.append([(start_node, next_up)] + p)
    for p  in self.get_paths_lattice(next_right, end_node):
      res.append([(start_node, next_right)] + p)
    return res

  def getPaths(self, s1, s2):
    """ Returns a set of candidate paths between state s1 and state s3.
    Arguments:
    - s1 : a State object
    - s2 : a State object
    """
    start_link = s1.link_id
    end_link = s2.link_id
    # This simple code does not handle that kind of cases
    if start_link == end_link:
      return []
    return [(s1, [start_link] + p + [end_link], s2) \
            for p in self.get_paths_lattice(start_link[1], end_link[0])]
