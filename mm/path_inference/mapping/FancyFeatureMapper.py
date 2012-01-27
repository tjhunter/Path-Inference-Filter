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
Created on Nov 22, 2011

@author: tjhunter
'''

from FeatureMapper import FeatureMapper
from utils import mm_dis
from mm.path_inference.json import decode_link_id, decode_LatLng
import math

def path_length(p):
  """ Returns the lenght of a path (meters). """
  res = 0.0
  for (latlng1, latlng2) in zip(p.latlngs[1:], p.latlngs[:-1]):
    res += mm_dis(latlng1, latlng2)
  return res

def num_stops(p, links):
  """ Returns the number of stops on a path.
  """
  return sum([links[lid]['stop'] for lid in p.links[:-1]])

def num_signals(p, links):
  """ Returns the number of red lights on a path.
  """
  return sum([links[lid]['signal'] for lid in p.links[:-1]])

def angle(lid1, lid2, links):
  """ Returns the angle (in radians) between two paths.
  """
  lla1 = decode_LatLng(links[lid1]['geom'][0])
  lla2 = decode_LatLng(links[lid1]['geom'][-1])
  llb1 = decode_LatLng(links[lid2]['geom'][0])
  llb2 = decode_LatLng(links[lid2]['geom'][-1])
  dxa = lla1.lat - lla2.lat
  dxb = llb1.lat - llb2.lat
  dya = lla1.lng - lla2.lng
  dyb = llb1.lng - llb2.lng
  na = math.sqrt(dxa * dxa + dya * dya)
  nb = math.sqrt(dxb * dxb + dyb * dyb)
  dxa /= na
  dya /= na
  dxb /= nb
  dyb /= nb
  return dxa * dxb + dya * dyb

def num_left_turns(p, links):
  """ The number of left turns on a path.
  """
  cosines = [angle(lid1, lid2, links) for (lid1, lid2) in zip(p.links[:-1], p.links[1:])]
  res = 0.0
  for c in cosines:
    if c > 0.2:
      res += 1
  return res

def num_right_turns(p, links):
  """ The number of right turns on a path.
  """
  cosines = [angle(lid1, lid2, links) \
             for (lid1, lid2) in zip(p.links[:-1], p.links[1:])]
  res = 0.0
  for c in cosines:
    if c < -0.2:
      res += 1
  return res

def average_tt(p, links):
  """ The average travel time on a path.
  """
  total_tt = 0.0
  # Sum over all the complete links:
  for lid in p.links:
    total_tt += links[lid]['length'] / links[lid]['speed_limit']
  total_tt -= p.start.offset / links[p.links[0]]['speed_limit']
  end_link_length = links[p.links[-1]]['length']
  total_tt -= (end_link_length - p.end.offset) \
    / links[p.links[-1]]['speed_limit']
  return total_tt

def average_speed(p, links):
  """ Average speed on a path.
  """
  l = path_length(p)
  if l < 1:
    return 0
  else:
    return average_tt(p, links) / path_length(p)

def max_lanes(p, links):
  """ Returns the maximum number of lanes on all links of a path.
  """
  return max([links[lid]['lanes'] for lid in p.links])

def min_lanes(p, links):
  """ Returns the mainimum number of lanes on all links of a path.
  """
  return min([links[lid]['lanes'] for lid in p.links])

class FancyFeatureMapper(FeatureMapper):
  """ Feature mapper class for fancy features.
  
  TODO: doc
  """
  NUM_PATH_FEATURES = 9
  NUM_STATE_FEATURES = 1
  # Some random buggy warning:
  def __init__(self, links):
    FeatureMapper.__init__(self)
    self.num_path_features = FancyFeatureMapper.NUM_PATH_FEATURES
    self.num_state_features = FancyFeatureMapper.NUM_STATE_FEATURES
    self.links = dict([(decode_link_id(link_dct['link_id']), link_dct) \
                       for link_dct in links])
  
  def path_features(self, path):
    return [path_length(path), num_stops(path, self.links), \
            num_signals(path, self.links), num_left_turns(path, self.links), \
            num_right_turns(path, self.links),  \
            average_tt(path, self.links),  average_speed(path, self.links),  \
            max_lanes(path, self.links), min_lanes(path, self.links)]
