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
Created on Sep 20, 2011

@author: tjhunter
'''

class PathBuilder(object):
  """ Creates candidate paths between states.
  
  INTERFACE ONLY.
  """
  
  def getPaths(self, s1, s2):
    """ Returns a set of candidate paths between state s1 and state s3.
    Arguments:
    - s1 : a State object
    - s2 : a State object
    """
    raise NotImplementedError()

  def getPathsBetweenCollections(self, sc1, sc2):
    trans1 = []
    trans2 = []
    paths = []
    n1 = len(sc1.states)
    n2 = len(sc2.states)
    num_paths = 0
    for i1 in range(n1):
      for i2 in range(n2):
        ps = self.getPaths(sc1.states[i1], sc2.states[i2])
        for path in ps:
          trans1.append((i1, num_paths))
          trans2.append((num_paths, i2))
          paths.append(path)
          num_paths += 1
    return (trans1, paths, trans2)
