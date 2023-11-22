#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

energy = np.loadtxt('./data/eng-matrix.txt')
energy = energy[:,0]

Emin = -5
Emax = 5
nE = 500
dE = (Emax - Emin) / (nE-1)
E = np.array([i*dE+Emin for i in range(nE)])

gE = np.zeros(nE)
for i in range(nE-1):
  idx = np.where(np.logical_and(energy>E[i], energy<E[i+1]))[0]
  gE[i] = np.sum(idx)


plt.figure()
plt.plot(E,gE)
plt.show()
plt.close()
