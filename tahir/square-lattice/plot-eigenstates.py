#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import glob

filepath = './data/'
stateslist = sorted(glob.glob( filepath+'eigenstate-phi-*.txt'))
#stateslist = np.loadtxt('./data/eigenstate-phi-049.txt')

for i in range(np.size(stateslist)):
  plt.figure(1000, figsize=(6,6))
  states = np.loadtxt(stateslist[i], dtype=complex)
  states = np.real( np.multiply( states, np.conj(states) ) )
  plt.plot(states[:,21*11//2-1], linewidth=0.75)
  plt.plot(states[:,21*11//2], linewidth=0.75)
  plt.plot(states[:,21*11//2+1], linewidth=0.75)
  #plt.show()
  fileout = './figures/eigenstate-phi-%03i.pdf' % (i)
  plt.savefig(fileout)
  plt.close(1000)
