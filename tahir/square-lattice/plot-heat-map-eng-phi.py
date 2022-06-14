#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob

# load configuration values
config = np.loadtxt('./data/config-floquet.txt')
rc = int(config[0])
mc = int(config[1])
phimin = config[2]
phimax = config[3]
nphi = int(config[4])
#phi = np.linspace( phimin, phimax, nphi )
phi = np.array([(i/nphi)**(1/2) for i in range(nphi)])*phimax
strnphi = str(nphi)

ns = 2*rc+1
m0 = (mc-0)*ns
mf = (mc+1)*ns

# load calculated values and states
energy = np.loadtxt('./data/eng-matrix.txt')
stateslist = sorted( glob.glob( './data/eigenstate-phi-*.txt') )
weight = np.zeros( (nphi, ns*(2*mc+1)) )
#weight = np.zeros( (mf-m0,nphi) )
#columns = np.arange(m0,mf,1)

# calculate weight/projection onto 0th order mode
for i, statefilename in enumerate(stateslist):
  if i==nphi:
    break
#  states = np.loadtxt( statefilename, dtype=complex)
  states = np.loadtxt( statefilename, dtype=complex, skiprows=m0, max_rows=mf )
  tmp = np.real( np.matmul( states.conj().T, states ) )
  weight[i,:] = np.diag( tmp , k=0 )
#  weight[i,:] = np.sum( tmp[m0:mf,:], axis=0 )
  if i == nphi-5:
    print(weight[i,:])
#Emax = np.max(energy[m0:mf,:])
#Emin = np.min(energy[m0:mf,:])
Emax = np.max(energy)
Emin = np.min(energy)
nE = 200
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
#    idx = np.where(np.logical_and(energy[m0:mf,i]>E[j],energy[m0:mf,i]<E[j+1]))[0]
    gE[i,j+1] = np.sum(weight[i,idx])
#    gE[i,j+1] = np.sum(idx)


fig, ax = plt.subplots(1,1)
img = ax.imshow( (np.flipud(gE.transpose()) ), interpolation='spline16', cmap='Blues', extent=[-1,1,-1,1])
#img = ax.imshow( (np.flipud(gE.transpose()) ), cmap='Blues', extent=[-1,1,-1,1])

xticks = np.linspace(-1,1,5, endpoint=True)
xlabelarray = np.linspace(phimin, phimax, 5, endpoint=True)**2
ax.set_xticks(xticks)
ax.set_xticklabels(['%1.2f' % val for val in xlabelarray])
ax.set_xlabel('$\phi_0^2$')

yticks = np.linspace(-1,1,5, endpoint=True)
ylabelarray = np.linspace(Emin, Emax, 5, endpoint=True)
ax.set_yticks(yticks)
ax.set_yticklabels(['%1.2f' % val for val in ylabelarray])
ax.set_ylabel('$E(\phi_0^2)$')

#fig.colorbar(img)
plt.savefig('./figures/dos.pdf', bbox_inches='tight')
