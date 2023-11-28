#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

energy = np.loadtxt('./data/eng-matrix.txt')
mc = 2
rc = 10
nr = 2*rc+1
energy = energy[mc*nr:(mc+1)*nr,:]
rows, columns = np.shape(energy)
# Find the y-shift
emin = np.min(energy)
energy -= emin
# See if the lowest energy is quadratic by fitting to ax^2
x = np.arange(0,columns)


plt.figure()
#plt.ylim(0, 0.02)
for i in range(rows):
  a = (energy[i,-1] - energy[i,0]) / (columns-1)**2
  y = x**2 * a + energy[i,0]
  plt.plot(x, energy[i,:], '.')
  plt.plot(x, y, 'k')
  #plt.plot(x,energy[i,:]-y)
  #print(np.max(np.abs(energy[i,:]-y)))
#plt.plot(x, y, '.', linewidth=1)

plt.show()
plt.close()
