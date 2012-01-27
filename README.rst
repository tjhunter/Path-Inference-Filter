Path inference filter quick documentation
==========================================


The path inference filter (PIF in short) maps GPS waypoints into trajectories
on a road network. It is used to map this sort of data:
http://youtu.be/OxCPL4KsDfI
into this sort of output: http://youtu.be/tj53gGCCNgs

This code is the academic code that goes along the paper: ...


This code is designed to be readable and correct, not fast. As such, you will
probably want to rewrite some core sections in your favourite programming
language. It is also extensively covered by a test suite that you can use as 
a reference. If you want a (much faster and more complete) implementation in 
scala and java, please contact the author.

Quick start guide
------------------

The python PIF uses the following libraries:
  
  - numpy >= 1.3
  
  - nose >= 1.5 (for testing only)

All you should need is::
  
  git clone git://github.com/tjhunter/Path-Inference-Filter
  cd Path-Inference-Filter
  nosetest

If all the diagnostic tests return correctly, you should be all set!


Basic filtering
----------------

A tutorial has been written to explain how to filter trajectories, which you 
can find in *mm/path_inference/example.py*.
The python PIF code does not include mapping or path discovery due to licensing
issues. You have to write your own code to interface the PIF with your favorite
data source and road network.


Learning
---------

No tutorial has been written for learning yet. However, the learning functions
should have enough documentation. You can take a look at the tests of the 
optimizer as a starting point: *mm/path_inference/learning_traj_optimizer.py*.