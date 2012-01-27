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
Created on Sep 16, 2011

@author: tjhunter
'''

class LatLng(object):
  """ A geolocation representation.
  """
  def __init__(self, lat, lng):
    self.lat = lat
    self.lng = lng

  def __eq__(self, other):
    return self.lat == other.lat and self.lng == other.lng

  def __ne__(self, other):
    return not self.__eq__(other)


class State(object):
  """ A state on the road network.
  """
  def __init__(self, link_id, offset, pos=None):
    """ 
     - link_id: a string for identifying the link.
     - offset: an offset on the link
     - pos: a LatLnt position
    """
    self.link_id = link_id
    self.offset = offset
    self.gps_pos = pos

  def __eq__(self, other):
    return self.link_id == other.link_id \
      and self.offset == other.offset \
      and self.gps_pos == other.gps_pos

  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __repr__(self):
    return "State[%s, %s]" % (str(self.link_id), str(self.offset))

class StateCollection(object):
  """
    id: a string identifier of the driver
    spots: a list of State objects.
    gps_pos: the LatLng gps observation.
    time: a DateTime object
  """
  def __init__(self, v_id, states, gps_pos, time):
    self.id = v_id
    self.states = states
    self.gps_pos = gps_pos
    self.time = time
  
  def __eq__(self, other):
      return (isinstance(other, self.__class__)
          and self.__dict__ == other.__dict__)

  def __ne__(self, other):
      return not self.__eq__(other)
  

class Path(object):
  """ A path object: represnts a path on the network.
  
  start_state: the start state
  end_state: the end state
  links: a sequence of link ids
  states: a list of states (optional, for drawing only)
  """
  def __init__(self, start_state, links, end_state, latlngs=None):
    self.start = start_state
    self.end = end_state
    self.links = links
    self.latlngs = latlngs

  def __eq__(self, other):
    return self.start == other.start \
      and self.end == other.end \
      and self.links == other.links

  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __repr__(self):
    return "Path[%s,links=%s,%s]" % (str(self.start), str(self.links), str(self.end))

class PathCollection(object):
  """ A collection of paths, and some additional informations.
  """
  def __init__(self, v_id, paths, start_time, end_time):
    self.id = v_id
    self.paths = paths
    self.start_time = start_time
    self.end_time = end_time
