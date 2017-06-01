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
""" Docstring.
"""
from mm.path_inference.learning_traj import LearningTrajectory

def simple_traj0():
  """ Simple trajectory.
  """
  features = [ [ [1.0], [-2.0] ],
               [ [-1.0] ],
             ]
  connections = [ [ (1, 0), (0, 0) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj1():
  """ Simple trajectory.
  """
  features = [ [ [1.0, 0.0], [-2.0, 0.0] ],
               [ [0.0, 1.0] ],
             ]
  connections = [ [ (1, 0), (0, 0) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj2():
  """ Simple trajectory, tests correct indices.
  """
  features = [ [ [0.0, 2.0] ],
               [ [2.0, 0.0], [-1.0, 0.0] ],
             ]
  connections = [ [ (0, 0), (0, 1) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj3():
  """ Simple trajectory with large features, 
  breaks non log-based implementations.
  """
  features = [ [ [0., 200.] ],
               [ [200., 0.], [-100., 0.] ],
             ]
  connections = [ [ (0, 0), (0, 1) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj4():
  """ Simple trajectory.
  """
  features = [ [ [1.0, 0.0], [-2.0, 0.0] ],
               [ [0.0, 1.0] ],
               [ [0.0, -1.0], [0.0, 2.0], [0.5, 0.5] ],
             ]
  connections = [ [ (1, 0), (0, 0) ],
                  [ (0, 0), (0, 1), (0, 2) ],
                ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj5():
  """ Simple trajectory.
  """
  features = [ [ [1.0], [-2.0] ],
               [ [-1.0] ],
               [ [-1.0], [2.0], [0.0] ],
             ]
  connections = [ [ (1, 0), (0, 0) ],
                  [ (0, 0), (0, 1), (0, 2) ],
                ]
  traj = LearningTrajectory(features, connections)
  return traj


def simple_traj6():
  """ Simple trajectory with more than 3 elements.
  """
  features = [ [ [1.0, 0.0], [-2.0, 0.0] ],
               [ [0.0, 1.0] ],
               [ [0.0, -1.0], [0.0, 2.0], [0.5, 0.5] ],
               [ [0.0, -1.0], [0.0, 2.0], [0.5, 0.5] ],
             ]
  connections = [ [ (1, 0), (0, 0) ],
                  [ (0, 0), (0, 1), (0, 2) ],
                  [ (1, 0), (2, 1), (0, 2) ],
                ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj7():
  """ Simple trajectory, with a disconnected point.
  This case should be properly handled.
  """
  features = [ [ [1.0], [-2.0] ],
               [ [-1.0], [-2.0] ],
             ]
  connections = [ [ (0, 0) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj8():
  """ Simple trajectory, with a single feature and feature values
  that are all zeros.
  Feature values are small enough to be safe against underflows.
  """
  features = [ [ [0.0], [0.0] ],
               [ [0.0], [0.0] ],
             ]
  connections = [ [ (0, 0), (0, 1), (1, 1) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj9():
  """ Simple trajectory, with a single feature and feature values
  that are all zeros.
  Feature values are small enough to be safe against underflows.
  """
  features = [ [ [1.0], [2.0] ],
               [ [1.0], [2.0] ],
             ]
  connections = [ [ (0, 0), (0, 1), (1, 1) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def simple_traj10():
  """ Simple trajectory, with a single feature and feature values
  that are very big. Should cause some underflow problems.
  """
  features = [ [ [1000.0, 0.0], [2000.0, 0.0] ],
               [ [500.0, 0.0], [3000.0, 0.0] ],
             ]
  connections = [ [ (0, 0), (0, 1), (1, 1) ] ]
  traj = LearningTrajectory(features, connections)
  return traj

def test_traj7():
  """ test_traj7 """
  traj = simple_traj7()
  assert traj.L == 2

def test_traj1():
  """ Simple test. """
  traj = simple_traj1()
  assert traj.L == 2

def test_traj2():
  """ Simple test 2. """
  traj = simple_traj2()
  assert traj.L == 2

if __name__ == '__main__':
  test_traj1()
