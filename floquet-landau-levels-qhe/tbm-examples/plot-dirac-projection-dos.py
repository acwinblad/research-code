#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob

# load configuration values
config = np.loadtxt('./data/config.txt')
rc = int(config[0])
mc = int(config[1])
h = float(config[2])
phimin = float(config[3])
phimax = float(config[4])
nphi = int(config[5])
#phi = np.array([(i/nphi)**(1/2) for i in range(nphi)])*phimax
strnphi = str(nphi)

nr = 2*rc+1
nm = 2*mc+1
m0 = (mc-0)*nr
mf = (mc+1)*nr

# load calculated values and states
energy = np.loadtxt('./data/eng-matrix.txt')
stateslist = sorted( glob.glob( './data/eigenstate-phi-*.txt') )

# calculate weight/projection on 0th order Block
weight = np.zeros( (nphi, 4*nr*nm) )
for i, statefilename in enumerate(stateslist):
  states = np.loadtxt( statefilename, dtype=complex, skiprows=m0, max_rows=4*nr )
  weight[i,:] = np.diag( np.real( np.matmul( states.conj().T, states ) ) , k=0 )

#wavg = np.average(weight)
#wstd = np.std(weight)
#threshold = wavg
#weight[weight<threshold] = 0
#weight[weight>=threshold] = 1

# calculate a weighted/projected density of states as a function of phi
Emax = np.max(energy)
Emin = np.min(energy)
Emax = +1.0*h
Emin = -1.0*h
nE = 400
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])

# place weighted eigenvalues in an energy box-bin (also a normal dos)
wE = np.zeros((nphi,nE))
gE = np.zeros((nphi,nE))
gausE = np.zeros((nphi,nE))
sigma = dE
sig2 = sigma**2
norm = (sigma*np.sqrt(np.pi))**(-1)
for i in range(nphi):
  #eidx = np.where(np.logical_and(energy[:,i]<Emax, energy[:,i]>Emin))
  gausE[i,0] = norm*np.sum(np.exp( -(E[0]-energy[:,i])**2 / sig2))
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
    gausE[i,j+1] = norm*np.sum(weight[i,:]*np.exp( -(E[j+1]-energy[:,i])**2 / sig2))
    wE[i,j+1] = np.sum(weight[i,idx])
    gE[i,j+1] = np.sum(np.size(idx))

#wE[wE!=0]=1
# setup plot for weighted density of states
fig, ax = plt.subplots(1,1)
phi = np.linspace(phimin,phimax,nphi)
eng = np.linspace(Emin,Emax,nE)
X, Y = np.meshgrid(phi, eng)

# set x-axis
xticks = np.linspace(phimin,phimax,5, endpoint=True)
xlabelarray = np.linspace( phimin, phimax, 5, endpoint=True)
ax.set_xticks(xticks)
ax.set_xticklabels(['%1.2e' % val for val in xlabelarray])
ax.set_xlabel('$\phi_E$')

# set y-axis
yticks = np.linspace(Emin,Emax,5, endpoint=True)
ylabelarray = np.linspace(Emin, Emax, 5, endpoint=True)
ax.set_yticks(yticks)
ax.set_yticklabels(['%1.2f' % val for val in ylabelarray])
ax.set_ylabel('$E(\phi_E)$')

# plot and save figures
img = ax.imshow( ( np.flipud(wE[1:].transpose()) )**(1.0), interpolation='nearest', cmap='Blues', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-projection.pdf', bbox_inches='tight')
plt.savefig('./figures/dos-projection.png', bbox_inches='tight')

# normal dos
img = ax.imshow( ( np.flipud(gE[1:].transpose()) )**(1.0), interpolation='nearest', cmap='Blues', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-full.pdf', bbox_inches='tight')

# gaussian dos
img = ax.imshow( ( np.flipud(gausE.transpose()) )**(1.0), interpolation='nearest', cmap='Blues', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-gaussian.pdf', bbox_inches='tight')
