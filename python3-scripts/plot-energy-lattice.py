#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

#np.set_printoptions(threshold='nan')

# solve in lattice space

plotSpectra = False
plotDOS     = True
filein = './data/%runfile%-tbr'
fileout = './figures/%runfile%-fig-test'
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
  n = np.size(energy)/4
  for i in range(2*n):
    plt.plot(xaxis, [energy[n+i],energy[n+i]], 'b', linewidth=0.05)
  plt.tight_layout()
  plt.savefig(fileout+'-energy-spectra.pdf')

if plotDOS==True:
  energyGap = np.loadtxt(filein+'-energy-band-gaps.txt')

  nE = 75
  nEG = 15 # needs to be odd

  Emax = energy[n]
  Emin = energy[3*n-1]

  dEgap = np.abs(energyGap[1]-energyGap[0])/nEG/2.
  dE = np.abs(Emax-energyGap[1])/nE/2.

  Eaxis = np.array([Emin+(2*j+1)*dE for j in range(nE)])
  Eaxis = np.append(Eaxis, np.array([energyGap[0]+(2*j+1)*dEgap for j in range(nEG)]))
  Eaxis = np.append(Eaxis, np.array([energyGap[1]+(2*j+1)*dE for j in range(nE)]))

  dos = np.zeros(np.size(Eaxis))
  for j in range(nE):
    idx = np.where(np.logical_and(energy<=Eaxis[j]+dE,energy>Eaxis[j]-dE))[0]
    dos[j] = np.size(idx)
  for j in range(nEG):
    idx = np.where(np.logical_and(energy<=Eaxis[j+nE]+dEgap, energy>Eaxis[j+nE]-dEgap))[0]
    dos[j+nE] = np.size(idx)
  for j in range(nE):
    idx = np.where(np.logical_and(energy<=Eaxis[j+nE+nEG]+dE,energy>Eaxis[j+nE+nEG]-dE))[0]
    dos[j+nE+nEG] = np.size(idx)

  plt.figure()
  plt.tight_layout()
  plt.grid("True")
  ymax = np.max(dos)
  plt.ylim(0,ymax)
  plt.ylabel('$g(\epsilon (t))$', fontsize=12)
  plt.xlabel('$\epsilon (t)$', fontsize=12)
  plt.plot([energyGap[0],energyGap[0]],[0,ymax], 'red', linewidth=0.5)
  plt.plot([energyGap[1],energyGap[1]],[0,ymax], 'red', linewidth=0.5)
  #plt.plot(Eaxis,dos/np.max(dos), 'black')
  plt.plot(Eaxis,dos, 'black', linewidth=0.5)
  #plt.show()
  plt.savefig(fileout+'-dos.pdf')

  plt.figure()
  plt.tight_layout()
  plt.grid("True")
  ymax = np.max(dos)*0.8
  plt.ylim(0,ymax)
  plt.xlim(energyGap[0]-2*dE,energyGap[1]+2*dE)
  plt.ylabel('$g(\epsilon (t))$', fontsize=12)
  plt.xlabel('$\epsilon (t)$', fontsize=12)
  plt.plot([energyGap[0],energyGap[0]],[0,ymax], 'red', linewidth=0.5)
  plt.plot([energyGap[1],energyGap[1]],[0,ymax], 'red', linewidth=0.5)
  #plt.plot(Eaxis,dos/np.max(dos), 'black')
  plt.plot(Eaxis,dos, 'black', linewidth=0.5)
  #plt.show()
  plt.savefig(fileout+'-dos-zoomed.pdf')
