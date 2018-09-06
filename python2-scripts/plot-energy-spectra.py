#!/usr/bin/python
#
# Created by: Aidan Winblad
#

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


# load in eigen- energy
energy = np.loadtxt('../../data/bdg-energy-triangle.txt')
n = np.size(energy)
energy = np.column_stack([energy,energy])
x = np.array([-1,1])

plt.figure(figsize=(2,4))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
plt.ylabel('$\epsilon (t)$', fontsize=12)
plt.xlim(-1,1)
plt.ylim(-6,6)
for i in xrange(n):
  plt.plot(x,energy[i,:], 'b', linewidth=0.25)
plt.tight_layout()
#plt.show()
plt.savefig('../data/fig-energy-spectra.pdf')
