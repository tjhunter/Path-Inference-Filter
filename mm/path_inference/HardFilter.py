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
Created on Nov 25, 2011

@author: tjhunter
'''

class HardFilter(object):
  """ Interface for a filter that performs hard assignments.
  
  Interesting fields:
   - traj : a trajectory object
   - assignments : a list of assignment (integer index, for each feature
       element of the trajectory)
  """
  # pylint: disable=w0201
  def __init__(self):
    pass
  
  def computeAssignments(self):
    """ Computes the assignments.
    """ 
    self.assignments = None
    raise NotImplementedError()
