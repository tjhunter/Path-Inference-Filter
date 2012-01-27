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
class PathCounter(object):
  """ Use it as a closure to count the paths.
  """
  def __init__(self):
    self.count = 0
    self.num_paths = []
  def call(self, dct):
    """ Closure call.
    """
    (_, paths, _, _) = dct
    self.num_paths.append(len(paths))
    self.count += 1
