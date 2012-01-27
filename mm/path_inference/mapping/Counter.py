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
from mm.path_inference.json import decode_StateCollection

class Counter(object):
  """ Provide a counter class for collecting statistics on the objects.
  """
  def __init__(self, filter_non_hired=False):
    self.count = 0
    self.num_states = []
    self.filter_non_hired = filter_non_hired
  def call(self, dct):
    ''' closure '''
    sc = decode_StateCollection(dct)
#    if self.filter_non_hired and sc.hired != True:
#      self.num_states.append(0)
#    else:
    self.num_states.append(len(sc.states))
    self.count += 1
    return None

