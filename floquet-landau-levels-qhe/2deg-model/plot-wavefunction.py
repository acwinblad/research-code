#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from pfapack import pfaffian as pf
import glob

filein = './data/'
config = np.loadtxt(filein+'config.txt')
rc = int(config[0])
mc = int(config[1])
h = float(config[2])
Efmin = float(config[3])
Efmax = float(config[4])
nEf = int(config[5])
strnEf = str(nEf)

energy = np.loadtxt(filein+'eng-matrix.txt')
stateslist = sorted( glob.glob(filein+'eigenstate-full-Ef-*.txt'))

nr = 2*rc+1
nm = 2*mc+1
m0 = (mc-0)*nr
mf = (mc+1)*nr

x = np.linspace(-rc,rc,nr)
for i, statefilename in enumerate(stateslist):
  states = np.loadtxt( statefilename )
  fig, ax = plt.subplots(1,1)
  ax.set_xlabel('x [a]')
  ax.set_ylabel('$|\psi|^2$')
  for j in range(nr):
    l = j+m0
    if( j%2 == 0):
      color = 'orange'
    else:
      color = 'purple'
    plt.plot(x, states[:,l]+0.01*j, color=color, zorder=-j)
    #plt.plot(states[:,l]+0.1*j, color=color, zorder=-j)
  plt.savefig('./figures/eigenstate-Ef-%03i.pdf' % (i), bbox_inches='tight')
  plt.close()

