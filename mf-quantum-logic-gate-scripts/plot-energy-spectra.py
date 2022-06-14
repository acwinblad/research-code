#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
#np.set_printoptions(threshold='nan')

# solve in lattice space

plotSpectra = True
plotDOS     = True
filein = './data/kitaev-triangle-chain'
#filein = './data/kitaev-triangle'
fileout = './data/figures/fig-kitaev-triangle'
energy = np.loadtxt(filein+'-energy.txt')

if plotSpectra==True:
  xaxis = np.array([-1,1])
  plt.figure(figsize=(2,4))
  plt.tick_params(
      axis='x',
      which='both',
      bottom='off',
      top='off',
      labelbottom='off')
  plt.ylabel('$\epsilon (t)$', fontsize=12)
  plt.xlim(-1.2,1.2)
  n = int(np.size(energy)/4)
  for i in range(2*n):
    plt.plot(xaxis, [energy[n+i],energy[n+i]], 'b', linewidth=0.05)
  plt.tight_layout()
  plt.savefig(fileout+'-energy-spectra.pdf')

if plotDOS==True:
  nE = 125

  Emax = energy[3*n]
  Emin = energy[n]

  dE = np.abs(Emax-Emin)/nE

  Eaxis = np.array([Emin+j*dE for j in range(nE)])

  dos = np.zeros(np.size(Eaxis))
  for j in range(nE):
    idx = np.where(np.logical_and(energy<=Eaxis[j]+dE,energy>=Eaxis[j]-dE))[0]
    dos[j] = np.size(idx)

  plt.figure()
  plt.tight_layout()
  plt.grid("True")
  ymax = np.max(dos)
  plt.xlim(Emin,Emax)
  plt.ylabel('$g(\epsilon (t))$', fontsize=12)
  plt.xlabel('$\epsilon (t)$', fontsize=12)
  plt.plot(Eaxis,dos/np.max(dos), 'black')
  #plt.show()
  plt.savefig(fileout+'-dos.pdf')
