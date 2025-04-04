#!/usr/bin/python
#
# bdg solver lattice p-wave triangle
# Created by: Aidan Winblad
# 09/07/2017
#
# This code solves the eigenvalue problem of the bdg matrix for a lattice p-wave equilateral triangle
# We should get an array of energy values. We will then wrap the energy values into a 
# energy density histogram. yada yada yada
#

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d
import string
import glob


def plot_individual_comparison(_idx, _triang, _energy, _states, _n):
  vmx = np.max(_states[0:_n/2,_idx])
  v = np.linspace(0,vmx,19)
  for i in _idx:
    # initialize plotting terms
    plt.figure()
    plt.xlabel('$x\ (a)$', fontsize=12)
    plt.ylabel('$y\ (a)$', fontsize=12)
    plt.tricontourf(_triang, _states[0:_n,i], v, cmap='viridis', vmin=0, vmaX=vmx) 
    clb = plt.colorbar(format='%1.4f')
    clb.ax.set_ylabel('$\|\Psi\|^2$',rotation=0,labelpad=15, fontsize=12)
    filename = '../../data/fig-wavefunction-energy-%02d.pdf' % (_n-1-i)
    plt.savefig(filename)
    plt.close()

def plot_individual(_triang, _energy, _states, _fileout):

  # initialize plotting terms
  vmx = np.max(_states)
  v = np.linspace(0,vmx,19)
  plt.figure()
  plt.xlabel('$x\ (a)$', fontsize=12)
  plt.ylabel('$y\ (a)$', fontsize=12)
  plt.title('$\epsilon = %0.4f$' % _energy)
  plt.tricontourf(_triang, _states[0:_n,i], v, cmap='viridis', vmin=0, vmax=vmx) 
  clb = plt.colorbar(format='%1.4f')
  clb.ax.set_ylabel('$\|\Psi\|^2$',rotation=0,labelpad=15, fontsize=12)
  plt.savefig(_fileout)
  plt.close()

def plot_subgroup(_idx, _triang, _energy, _states, _n, _name):
  vmx = np.max(_states[0:_n,_idx])
  v = np.linspace(0,vmx,19)
  f, axes = plt.subplots(2, 2, sharex='col', sharey='row')
  # initialize plotting terms
  f.text(0.45,0.04, '$x\ (a)$', ha='center')
  f.text(0.04,0.5, '$y\ (a)$', ha='center', rotation='vertical')
  f.subplots_adjust(left=0.125,right=0.75)
  i=0
  for ax in axes.flat:
    im =  ax.tricontourf(_triang, _states[0:_n,_idx[i]], v, cmap='viridis', vmin=0, vmax=vmx) 
    ax.set(adjustable='box-forced',aspect='equal')
    i+=1
  for j, ax in enumerate(axes.flat):
    ax.text(0.1, .9, string.ascii_uppercase[j], transform=ax.transAxes, size=12, weight='bold')
  
  cbar_ax = f.add_axes([0.775, 0.12, 0.03, 0.75])
  clb = f.colorbar(im, cax=cbar_ax,format='%1.4f')
  clb.ax.set_ylabel('$\|\Psi\|^2$',rotation=0,labelpad=15, fontsize=12)

  filename = '../../data/fig-wavefunction-energy-%s.pdf' % _name
  plt.savefig(filename)
  plt.close()

# create triangular mesh
#filein = '../../data/batchruns/data/0001-tbr'
filepath = '../../data/'
x,y = np.loadtxt(filepath+'0001-tbr-coordinates.txt', unpack=True)
n = np.size(x)
triang = mtri.Triangulation(x,y)
# load in eigen- energy and states
energylist = sorted(glob.glob(filepath+'*-tbr-edge-state-energy.txt'))
stateslist = sorted(glob.glob(filepath+'*-tbr-edge-state-states.txt'))
plot_edge_states = True

# load in the energy band gap limits
# create index array of which energies lie in that range
if plot_edge_states==True:
  
  for i in range(np.size(energylist)): 
    edgeEnergies = np.loadtxt(energylist[i])
    n = np.size(edgeEnergies)
    if n > 0 :
     states = np.loadtxt(stateslist[i])
     print np.shape(states)
      #plot_individual(triang, energy, states, fileout)

  


