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
Created on Sep 20, 2011

@author: tjhunter

Encoding/decoding conversions.
'''
from structures import LatLng, State, Path, StateCollection
import datetime

def encode_LatLng(gps):
  return {'lat' : gps.lat, 'lng' : gps.lng}

def decode_LatLng(dct):
  return LatLng(dct['lat'], dct['lng'])

def encode_link_id(link_id):
  (nid, direction) = link_id
  return {'id': nid, 'direction': direction}

def decode_link_id(dct):
  return (dct['id'], dct['direction'])

def encode_State(state):
  return {'link':encode_link_id(state.link_id), \
          'offset':state.offset,\
          'gps_pos':encode_LatLng(state.gps_pos)}

def decode_State(dct):
  gps_pos = decode_LatLng(dct['gps_pos']) if 'gps_pos' in dct else None
  return State(decode_link_id(dct['link']), \
               dct['offset'], gps_pos)

def encode_Path(path):
  return {'start':encode_State(path.start), \
          'links':[encode_link_id(link_id) for link_id in path.links], \
          'end':encode_State(path.end), \
          'latlngs':[encode_LatLng(latlng) for latlng in path.latlngs]}

def decode_Path(dct):
  latlngs = [decode_LatLng(dct2) for dct2 in dct['latlngs']] \
    if 'latlngs' in dct else None
  return Path(decode_State(dct['start']), \
              [decode_link_id(dct2) for dct2 in dct['links']], \
              decode_State(dct['end']), \
              latlngs)

def encode_time(time):
  return {'year':time.year, \
          'month':time.month, \
          'day':time.day, \
          'hour':time.hour, \
          'minute':time.minute, \
          'second':time.second}

def decode_time(dct):
  return datetime.datetime(dct['year'], dct['month'], dct['day'], \
                           dct['hour'], dct['minute'], dct['second'])

def encode_StateCollection(sc):
  return {'id':sc.id, 'latlng':encode_LatLng(sc.gps_pos), \
          'time':encode_time(sc.time), \
          'states': [encode_State(state) for state in sc.states]}

def decode_StateCollection(dct):
  return StateCollection(dct['id'], [decode_State(dct2) \
                                     for dct2 in dct['states']], \
                         decode_LatLng(dct['latlng']), \
                         decode_time(dct['time']))

