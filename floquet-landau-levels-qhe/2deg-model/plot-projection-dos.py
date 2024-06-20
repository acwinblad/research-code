#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob
from scipy.stats import gaussian_kde

# load configuration values
mc = 5
rc = 5
filepath = './data/'
#config = np.loadtxt(filepath+'config.txt')
config = np.loadtxt('./data/config.txt')
rc = int(config[0])
mc = int(config[1])
t = float(config[2])
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
energy = np.loadtxt(filepath+'eng-matrix.txt')
stateslist = sorted( glob.glob( filepath+'eigenstate-phi-*.txt') )
bottombandproj = np.loadtxt(filepath+'bottom-band-projector.txt')

# calculate weight/projection on 0th order Block
weight = np.zeros( (nphi, nr*nm) )
bottomweight = np.zeros( (nphi, nr*nm) )
for i, statefilename in enumerate(stateslist):
  #states = np.loadtxt( statefilename, dtype=complex, skiprows=m0, max_rows=nr )
  states = np.loadtxt( statefilename, dtype = complex)
  tmp = np.real(np.matmul( states.conj().T, states ))
  tmp2 = np.real(np.matmul( states.conj().T, bottombandproj))
  tmp2 = np.real(np.matmul( tmp2, states))
  #print(tmp)
  weight[i,:] = np.diag( tmp, k=0 )
  bottomweight[i,:] = np.diag( tmp2, k=0 )
  #weight[i,:] = np.diag( states, k=0 )

#wavg = np.average(weight)
#weight = weight[0::4,:]
#weight[:,1::4] = 0
#weight[:,2::4] = 0
#weight[:,3::4] = 0

#wstd = np.std(weight)
#threshold = wavg
#weight[weight<threshold] = 0
#weight[weight>=threshold] = 1

# calculate a weighted/projected density of states as a function of phi
Emax = -3.0*t
Emin = -4.0*t
nE = 400
dE = (Emax - Emin)/(nE-1)
#E = np.array([i*dE+Emin for i in range(nE)])
E = np.arange(0,nE)*dE+Emin

# place weighted eigenvalues in an energy box-bin (also a normal dos)
wE = np.zeros((nE,nphi))
bwE = np.zeros((nE,nphi))
gE = np.zeros((nE,nphi))
gausgE = np.zeros((nE,nphi))
gauswE = np.zeros((nE,nphi))
gausbwE = np.zeros((nE,nphi))
sigma = dE
sig2 = sigma**2
norm = (sigma*np.sqrt(np.pi))**(-1)

for i in range(nphi):
  #eidx = np.where(np.logical_and(energy[:,i]>=Emin, energy[:,i]<=Emax))[0]
  gausgE[0,i] = norm*np.sum(np.exp( -(E[0] - energy[:,i])**2 / sig2))
  gauswE[0,i] = norm*np.sum(weight[i,:]*np.exp( -(E[0] - energy[:,i])**2 / sig2))
  gausbwE[0,i] = norm*np.sum(bottomweight[i,:]*np.exp( -(E[0] - energy[:,i])**2 / sig2))
  for j in range(nE-1):
    #idx = np.where(np.logical_and(energy[eidx,i]>E[j],energy[eidx,i]<E[j+1]))[0]
    idx = np.where(np.logical_and(energy[:,i]>=E[j],energy[:,i]<E[j+1]))[0]
    wE[j+1, i] = np.sum(weight[i,idx])
    bwE[j+1, i] = np.sum(bottomweight[i,idx])
    gE[j+1, i] = np.size(idx)
    gausgE[j+1, i] = norm*np.sum(np.exp( -(E[j+1] - energy[:,i])**2 / sig2))
    gauswE[j+1, i] = norm*np.sum(weight[i,:]*np.exp( -(E[j+1] - energy[:,i])**2 / sig2))
    gausbwE[j+1, i] = norm*np.sum(bottomweight[i,:]*np.exp( -(E[j+1] - energy[:,i])**2 / sig2))

#wE[wE!=0]=1
# setup plot for weighted density of states
fig, ax = plt.subplots(1,1)
x = np.linspace(phimin,phimax,nphi//2)
phi = np.linspace(phimin,phimax,nphi)
Eng = np.linspace(Emin,Emax,nE)
y = 0.9*x**1.5 +0.191*2
X, Y = np.meshgrid(phi,Eng)

# set x-axis
xticks = np.linspace(phimin,phimax,5, endpoint=True)
xlabelarray = np.linspace( phimin, phimax, 5, endpoint=True)
ax.set_xticks(xticks)
ax.set_xticklabels(['%1.2f' % val for val in xlabelarray])
ax.set_xlabel('$\phi_0$')

# set y-axis
yticks = np.linspace(Emin,Emax,5, endpoint=True)
ylabelarray = np.linspace(Emin, Emax, 5, endpoint=True)
ax.set_yticks(yticks)
ax.set_yticklabels(['%1.2e' % val for val in ylabelarray])
ax.set_ylabel('$Energy\ (eV)$')

# plot and save figures
# normal dos
img = ax.imshow( gE[1:]**(1.0), origin='lower', interpolation='spline16', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-full.pdf', bbox_inches='tight')

# proj dos
img = ax.imshow( wE[1:]**(1.0), origin='lower', interpolation='nearest', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-projection.pdf', bbox_inches='tight')

# proj bottom dos
img = ax.imshow( bwE[1:]**(1.0), origin='lower', interpolation='nearest', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-projection-bottom.pdf', bbox_inches='tight')

# gaussian dos
img = ax.imshow( gausgE**(1.0), origin='lower', interpolation='spline16', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-full-gaussian.pdf', bbox_inches='tight')

# projected gaussian dos
img = ax.imshow( gauswE**(1.0), origin='lower', interpolation='spline16', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-projection-gaussian.pdf', bbox_inches='tight')

# projected gaussian bottom dos
img = ax.imshow( gausbwE**(1.0), origin='lower', interpolation='spline16', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-projection-gaussian-bottom.pdf', bbox_inches='tight')

