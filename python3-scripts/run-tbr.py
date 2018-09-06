#!/usr/bin/python

import os
import numpy as np

runmom  = True
runlat  = True
runplot = True
run = np.int('%runfile%')

if runmom == True:
  os.system('python3 ./template-tbr-momentum.auto-gen.%i.py' % run)
if runlat == True:
  os.system('python3 ./template-tbr-lattice.auto-gen.%i.py' % run)
