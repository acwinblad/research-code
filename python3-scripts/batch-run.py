#!/usr/bin/python

import os
import numpy as np


# This batch run will consist of the following values
runfile = 'batchrun01-'
Vx = 0.
#Vy = np.array([0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0])
Vy = np.array([10])
Vz = 1000.
delta = 10.
t = 1.
mufctr = 3.
alphafctr = 0.25
nrmom = 500
nrlat = 50

for i in range(np.size(Vy)):
  runnum = '%04i' % i
  os.system('python3 ./tbm-momentum-slice.py %s %f %f %f %f %f %f %f %i' % (runfile+runnum, Vx, Vy[i], Vz, delta, t, mufctr, alphafctr, nrmom))
  #os.system('python3 ./tbm-momentum-rashba-in-plane-magnetic.py %s %f %f %f %f %f %f %f %i' % (runfile+runnum, Vx, Vy[i], Vz, delta, t, mufctr, alphafctr, nrmom))

#for i in range(np.size(Vy)):
#  os.system('python3 ./tbm-lattice-rashba-in-plane-magnetic.py %s %f %f %f %f %f %f %f %i' % (runfile+runnum, Vx, Vy[i], Vz, delta, t, mufctr, alphafctr, nrlat))
