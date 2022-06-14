#!/usr/bin/python3
#
# wavefunction plotting software for an equilateral triangle base
# Created by: Aidan Winblad
# 08/30/2021
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_individual(_idx, _triang, _energy, _states, _n):
  for i in _idx:
    vmx = np.max(_states[0:_n,i])
    v = np.linspace(0,vmx,19)
    plt.figure()
    plt.xlabel('$x\ (a)$', fontsize=12)
    plt.ylabel('$y\ (a)$', fontsize=12)
    plt.title('$\epsilon$ = %1.4e' % _energy[i])
    #plt.tricontourf(_triang, _states[0:_n,i], v, cmap='viridis', vmin=0, vmax=vmx)
    if i<n :
      plt.tricontourf(_triang, _states[0:_n,i], cmap='Blues')
    else :
      plt.tricontourf(_triang, _states[_n:2*_n,i], cmap='Blues')
      #plt.tricontourf(_triang, _states[0:_n,i], cmap='viridis')
    clb = plt.colorbar(format='%1.4f')
    clb.ax.set_ylabel('$\|\Psi\|^2$' , rotation=0, labelpad=15, fontsize=12)
    filename = './data/figures/fig-wavefunction-energy-%02d.pdf' % (i+1)
    plt.savefig(filename)
    plt.clf()

#filein = './data/kitaev-triangle'
#filein = './data/kitaev-triangle-chain'
filein = './data/hollow-triangle'
x,y = np.loadtxt(filein+'-coord.txt', unpack=True)
n = np.size(x)
triang = mtri.Triangulation(x,y)
energy = np.loadtxt(filein+'-energy.txt')
states = np.loadtxt(filein+'-states.txt')

#idxuser = np.arange(0,2*n)
idxuser = np.arange(n-4,n+4)
#idxuser = np.arange(0,10)

plot_individual(idxuser, triang, energy, states, n)

