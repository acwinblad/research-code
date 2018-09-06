#!/usr/bin/python
#
# Created by: Aidan Winblad
#

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


# load in eigen- energy
energy = np.loadtxt('../../data/bdg-energy-triangle.txt')
nbins = 30

gE, E = np.histogram(np.abs(energy),bins=10,range=(0,np.round(np.max(energy))))
gE, E = np.histogram(energy,bins=nbins)
plt.xlabel('$\epsilon (t)$', fontsize=12)
plt.ylabel('$g(\epsilon)/g_{max}$', fontsize=12)
plt.xlim(np.min(E),np.max(E))
plt.plot(E[1:]-np.abs(E[0]-E[1])/2.,gE)
plt.show()
#plt.savefig('../data/fig-dos-real-space.pdf')
