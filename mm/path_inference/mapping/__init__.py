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
""" Mapping data package.
"""
# Standard imports
from Counter import Counter
try:
  from FancyFeatureMapper import FancyFeatureMapper
  from FeatureMapper import FeatureMapper
except ImportError:
  pass
from HFDecimation import HFDecimation
from FlowAnalysis import FlowAnalysis
from PathCounter import PathCounter
from SparseDecimation import SparseDecimation
