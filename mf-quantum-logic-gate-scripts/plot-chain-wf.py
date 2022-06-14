#!/usr/bin/python3
#
# wavefunction plotting software for an equilateral triangle base
# Created by: Aidan Winblad
# 08/30/2021
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

filein = './data/kitaev-triangle-chain'
energy = np.loadtxt(filein+'-energy.txt')
n = np.size(energy)//4
states = np.loadtxt(filein+'-states.txt')

j = 2*n-1
plt.figure()
plt.tight_layout()
plt.grid("True")
plt.title('$\epsilon$ = %1.4e' % energy[j])
plt.plot(states[0:n,j])
plt.plot(states[2*n:3*n,j+1])
plt.show()
