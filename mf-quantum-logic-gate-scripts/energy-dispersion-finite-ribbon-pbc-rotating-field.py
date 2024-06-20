#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PI = np.pi

# System parameters
t = 1
delta = t
mu = 1.6*t
a = 1
n = 50
pbc = False

# Vector potential strength
A = 3.00 / a

# Zero out the BdG matrix
bdg = np.zeros((2*n,2*n),dtype='complex')

# Create angle arrays and energy arrays
nE = 4
nt = 8*90
ntp = 5*nt//6
ntq = nt//6
tf = 2*PI
tvals = np.linspace(0,tf,nt)
evt = np.zeros((2*nE,nt))

# The order parameter will not change so initialize once
bdg[0:n,n:2*n] = delta * (np.diag(np.ones(n-1),k=-1) - np.diag(np.ones(n-1),k=1))
if(pbc):
  bdg[0,2*n-1] = delta
  bdg[n-1,n] = -delta

for k, angle in enumerate(tvals):
  # Construct the BdG Hamiltonian for varying vector potential angles
  phi = -a*A*np.sin(angle)
  bdg[0:n,0:n] = -t * (np.exp(1.0j * phi) * np.diag(np.ones(n-1),k=1) + np.exp(-1.0j * phi) * np.diag(np.ones(n-1),k=-1)) - mu * np.eye(n)
  if(pbc):
    bdg[0,n-1] = -t * np.exp(-1.0j * phi)
    bdg[n-1,0] = -t * np.exp(1.0j * phi)
  bdg[n:2*n,n:2*n] = -bdg[0:n,0:n].T

  # Solve the eigenvalue problem for energies only
  eng = np.linalg.eigvalsh(bdg, UPLO = 'U')

  evt[:,k] = eng[n-nE:n+nE]

evtp = np.hstack([evt[:,ntp:], evt[:,0:ntp]])
evtq = np.hstack([evt[:,ntq:], evt[:,0:ntq]])

plt.figure()
xf = 1/6
xvals = tvals/(2*PI)
plt.rcParams.update({'font.size':13})
plt.xlim(xvals[0],xvals[0]+xf)
plt.xlabel(r"$\varphi$ ($2\pi$)", fontsize=18)
plt.xticks(np.linspace(xvals[0],xvals[0]+xf,5))
plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))
plt.ylabel('Energy (t)', fontsize=18)
#plt.ylim(-0.005,.005)
plt.ylim(-0.2,.2)

for i in range(nE):
  plt.plot(xvals,evt[i,:], 'C0')
  plt.plot(xvals,evt[nE+i,:], 'C0', label='_nolegend_')
  plt.plot(xvals,evtp[i,:], 'C1:')
  plt.plot(xvals,evtp[nE+i,:], 'C1:', label='_nolegend_')
  plt.plot(xvals,evtq[i,:], 'C2--')
  plt.plot(xvals,evtq[nE+i,:], 'C2--', label='_nolegend_')
#plt.ylim(-0.1,0.1)
plt.legend(['bottom edge','left edge','right edge'], loc='upper right')
plt.tight_layout()
plt.savefig('./data/figures/energy-dispersion-finite-ribbon-pbc-rotating-field.pdf')
#plt.show()
plt.close()

