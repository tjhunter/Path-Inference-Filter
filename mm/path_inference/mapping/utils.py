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
Created on Nov 8, 2011

@author: tjhunter
'''
from mm.path_inference.utils import distance as mm_dis # Name clash with pylab

def cut_points_by_distance(scs, n=10, max_distance=45, \
                           min_distance=1.0, min_length=10):
  """ Cuts points into tracks according to some distance criterion.
  Arguments:
   - scs: a list of StateCollection items corresponding to a run
   - n: index shift, > 0. The algorithm will compare the distance between
      sc[i] and sc[i+n]. If this distance is found to be greater than
      n * max_distance, the trjactory is cut at i.
   - max_distance: the maximum distance between two points. Two points
     separated by more than this distance are cut.
   - min_distance: minimum distance between two point. Below this distance,
     the new point is discard for beeing too close.
   - min_length: the minimum length of a track.
  n: 
  """
  # Meters, corresponds to 1-second probe data
  N = len(scs)
  # Compute the n-cut distance first to get an idea of long-terms cuts
  dsn = [mm_dis(scs[i].gps_pos, scs[i+n].gps_pos) for i in xrange(N-n)]
  # Distance cuts:
  d_cuts = [i for i in xrange(N-n) if dsn[i] > n*max_distance]
  points = [0] + d_cuts + [N]
  res = []
  for (start, end) in zip(points[:-1], points[1:]):
    all_points = scs[start:end]
    all_points.reverse()
    l = [all_points[-1]]
    while all_points:
      x = all_points.pop()
      if mm_dis(x.gps_pos, l[-1].gps_pos) > min_distance:
        l.append(x)
    if len(l) >= min_length:
      res.append(l)
  return res

def remove_spurious_points(scs, n=3, max_distance=45, \
                           min_distance=1.0, min_length=10):
  """ Removes a number of points that are of no interest or may cause trouble:
  - points that correspond to no move (stationary vehicle)
  - far points that correspond to GPS errors that make the trajectory jump
  Arguments:
  scs: a list of StateCollection items corresponding to a run
  n: the maximum number of points that may group as a GPS error
  max_distance: that maximum distance between points corresponding to a valid
    travel. Over this distance, the point is assumed to be disconnected.
  min_distance: under this distance, the vehicle is assumed to be stationary
  min_length: minimum size of groups of points to correspond to a valid
    trajectory chunk.
  
  Returns: a list of list of valid tracks.
  """
  res = []
  # This is a queue that will contain the points
  # Each point will be popped out and compared to the head of the 
  # current track.
  all_scs = list(scs)
  all_scs.reverse() # Need to reverse, since we will pop() from the start.
  current_track = []
  current_junk = []
  while all_scs:
    current_sc = all_scs.pop()
    if not current_track:
      current_track = [current_sc]
    else:
      last_sc = current_track[-1]
      assert last_sc.time < current_sc.time
      # Compare the distance
      d = mm_dis(last_sc.gps_pos, current_sc.gps_pos)
      # If too far, add to the junk
      if d < min_distance:
        continue
      if d > max_distance:
        current_junk.append(current_sc)
        # If junk if full, discard it, cut the track and start again
        if len(current_junk) > n:
          current_junk = []
          # If the cut track is long enough, keep it
          if len(current_track) >= min_length:
            res.append(current_track)
          current_track = []
      else:
        current_track.append(current_sc)
  # And take care of the last track
  if len(current_track) >= min_length:
    res.append(current_track)
  return res


def get_track_units(flow_units, path_counts, state_counts, min_path_count=10, \
                    min_state_count=1, min_traj_unit_length=5, \
                    max_traj_unit_length=3000):
  """ Computes the split of the track into trajectories that are
  self-coherent (every point is reachable and has a decent number
  of states).
  flow_units: a list of list of indexes in the track, each list of index is
    self-coherent
  path_counts: the number of paths between a pair of points.
  state_counts: list of integers, counts the number of projections for each 
    gps point
  
  The current implementation is a bit loose and does not try to get as many 
  points as possible. It does not really matter to loose a few points here 
  and there.
  """
  track_length = min(len(state_counts), len(path_counts))
  # The states that are marked as cut states.
  black_list = set([i for i in range(track_length) \
                    if (state_counts[i] < min_state_count)])
  for i in range(1, track_length):
    if path_counts[i] < min_path_count:
      black_list.add(i-1)
      black_list.add(i)
  for flow_unit in flow_units:
    if flow_unit[-1] >= track_length:
      if flow_unit[0] >= track_length:
        break
      flow_unit = [i for i in flow_unit if i < track_length]
      if not flow_unit:
        continue
    black_list.add(flow_unit[0])
    black_list.add(flow_unit[-1])
  black_list.add(0)
  black_list.add(track_length-1)
  # Now we have all the bad points, find all the good ones.
  bad_points = list(black_list)
  bad_points.sort()
  cut_pairs = zip(bad_points[:-1], bad_points[1:])
  res = []
  def chunks(l, n):
    for idx in xrange(0, len(l), n):
        yield l[idx:idx+n]
  
  for (x, y) in cut_pairs:
    if y-x >= min_traj_unit_length:
      for chunk in chunks(range(x, y), max_traj_unit_length):
        if len(chunk) >= min_traj_unit_length:
          res.append(chunk) 
  return res

