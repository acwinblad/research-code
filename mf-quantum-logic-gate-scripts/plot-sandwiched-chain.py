#!/usr/bin/python3
#
# wavefunction plotting software for an equilateral triangle base
# Created by: Aidan Winblad
# 08/30/2021
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

filein = './data/double-chain'
energy = np.loadtxt(filein+'-energy.txt')
n = np.size(energy)//2
n1 = n//3
states = np.loadtxt(filein+'-states.txt')

fig, ax = plt.subplots()
plt.grid("True")
plt.ylabel('$|\psi|^2$', fontsize=14)
plt.xlabel('x (a)', fontsize=14)
#plt.title('$\mu$ = -0.5t, B = 0.015$\pi$', fontsize=14)
m = 2*1
color = iter(plt.cm.rainbow(np.linspace(0,1,m)))
for i in range(m):
  #plt.plot(states[:,n-m//2+i])
  label = '$\epsilon$ = %1.2e' % energy[n-m//2+i]
  c = next(color)
  ax.plot(states[:n,n-m//2+i]+states[n:2*n,n-m//2+i]+0.01*(m-1-i), c=c, label=label)
legend = ax.legend(loc='center right', shadow=True, fontsize='large')
plt.savefig('./data/figures/double-chain-mu-n0_5-B-0_015pi.pdf')
plt.show()
plt.close()
