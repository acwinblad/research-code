#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob

# load configuration values
config = np.loadtxt('./data/config-floquet.txt')
rc = int(config[0])
mc = int(config[1])
phimin = config[2]
phimax = config[3]
nphi = int(config[4])
phi = np.array([(i/nphi)**(1/2) for i in range(nphi)])*phimax
strnphi = str(nphi)

ns = 2*rc+1
m0 = (mc-0)*ns
mf = (mc+1)*ns

# load calculated values and states
energy = np.loadtxt('./data/eng-matrix.txt')

# calculate a density of states as a function of phi
Emax = np.max(energy)
Emin = np.min(energy)
nE = 200
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])

# place weighted eigenvalues in an energy box-bin
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
    gE[i,j+1] = np.sum(idx)

# setup plot for density of states
fig, ax = plt.subplots(1,1)

# set x-axis
xticks = np.linspace(-1,1,5, endpoint=True)
xlabelarray = np.linspace(phimin, phimax, 5, endpoint=True)**2
ax.set_xticks(xticks)
ax.set_xticklabels(['%1.2f' % val for val in xlabelarray])
ax.set_xlabel('$\phi_0^2$')

# set y-axis
yticks = np.linspace(-1,1,5, endpoint=True)
ylabelarray = np.linspace(Emin, Emax, 5, endpoint=True)
ax.set_yticks(yticks)
ax.set_yticklabels(['%1.2f' % val for val in ylabelarray])
ax.set_ylabel('$E(\phi_0^2)$')

# plot and save figures
img = ax.imshow( (np.flipud(gE.transpose()) )**2, interpolation='spline16', cmap='binary', extent=[-1,1,-1,1])
plt.savefig('./figures/dos-v0.pdf', bbox_inches='tight')
