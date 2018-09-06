#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

#np.set_printoptions(threshold='nan')

# solve in lattice space

run = 0
filename = '../../data/tbm-lattice-rashba-in-plane-magnetic-run-%s' % run
energy = np.loadtxt(filename+'-energy.txt')
energyGap = np.loadtxt(filename+'-energy-band-gaps.txt')
idx = np.where(np.logical_and(energy>energyGap[0],energy<energyGap[1]))[0]
edgeEnergy = energy[idx]
zeroEnergy = np.average(energyGap[0:1])
deltaEdgeEnergy = edgeEnergy-zeroEnergy
xaxis = np.array([i*0 for i in range(np.size(deltaEdgeEnergy))])

figname = '../../data/fig-tbm-lattice-rashba-in-plane-magnetic-run-%s' % run

plt.figure()
plt.tight_layout()
plt.grid("True")
plt.xlim(-1,1)
plt.ylabel('$\delta\epsilon_0 (t)$', fontsize=12)
plt.xlabel('$V_y [run]$', fontsize=12)
plt.scatter(xaxis,deltaEdgeEnergy)
plt.savefig(figname+'-spectral-flow.pdf')
