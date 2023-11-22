#/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

energy = np.loadtxt('./landau-levels.txt')
config = np.loadtxt('./config-landau-levels.txt')
phimin = config[0]
phimax = config[1]
nphi = int(config[2])
phi0 = np.linspace(phimin, phimax, nphi)

Emin = 3.99-0.001
Emax = 4.00001+0.001
nE = 250
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j], energy[:,i]<E[j+1]))[0]
    gE[i, j+1] = np.size(idx)


fig, ax = plt.subplots(1,1)
img = ax.imshow(np.flipud(gE.transpose()), interpolation='spline16', cmap='binary', extent=[-1,1,-1,1])

plt.title('Landau levels')
plt.savefig('./fig-landau-levels-is.pdf')
