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
Created on Nov 27, 2011

@author: tjhunter
'''
from mm.path_inference.HardFilter import HardFilter

class SoftFilter(HardFilter):
  """ Interface for a filter that performs soft assignments (probability
  distributions).
  
  Interesting fields:
   - traj : a trajectory object
   - probabilities : list of numpy arrays
   - log_probabilities : list of numpy arrays
   - assignments : the most likely element, computed from the probabilities.
  """
  # pylint: disable=w0201
  def __init__(self):
    HardFilter.__init__(self)
  
  def computeProbabilities(self):
    """ This function computes the values of self.probabilities 
    and self.log_probabilities.     
    """
    self.probabilities = None
    raise NotImplementedError()
  
  def computeAssignments(self):
    self.assignments = [list(probs).index(max(probs)) for \
                        probs in self.probabilities]
