#!/usr/bin/python3
#
# wavefunction plotting software for an equilateral triangle base
# Created by: Aidan Winblad
# 08/30/2021
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

filein = './data/kitaev-peierls-chain'
energy = np.loadtxt(filein+'-energy.txt')
n = np.size(energy)//2
states = np.loadtxt(filein+'-states.txt')

plt.figure()
plt.tight_layout()
plt.grid("True")
plt.title('$\epsilon$ = %1.4e, %1.4e' % (energy[n-1], energy[n]))
plt.plot(states[0:n,n-2]+states[n:2*n,n-2])
plt.plot(states[0:n,n-1]+states[n:2*n,n-1])
plt.plot(states[0:n,n]+states[n:2*n,n])
plt.plot(states[0:n,n+1]+states[n:2*n,n+1])
plt.show()
