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
n1 = n//2+1
n2 = n1-2
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
  botChainC = np.append(states[n1-1:n,n-m//2+i], states[0,n-m//2+i])
  botChainCDag = np.append(states[n+n1-1:2*n,n-m//2+i], states[n,n-m//2+i])
  ax.plot(states[:n1,n-m//2+i]+states[n:n+n1,n-m//2+i]+(1.0+0.02*i), c=c, label=label)
  #ax.plot(np.flipud(states[n1-1:n,n-m//2+1]+states[n+n1-1:2*n,n-m//2])+0.02*i, c=c)
  ax.plot(np.flipud(botChainC+botChainCDag)+0.02*i, c=c)
legend = ax.legend(loc='center right', shadow=True, fontsize='large')
#plt.savefig('./data/figures/double-chain-mu-p1_1-B-0_16pi.pdf')
plt.show()
plt.close()
