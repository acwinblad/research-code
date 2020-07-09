#!/usr/bin/python3

import numpy as np

def hami(_i, _phi0, _ka):
  return 2*np.cos(2*np.pi*_phi0*_i - _ka) + 2

rc = 10
ns = 2*rc+1

ka = 0.1
phimin = 1e-4
phimax = 0.001
nphi = 250
phi = np.linspace(phimin, phimax, nphi)

energy = np.zeros( (ns, nphi) )
hn = np.zeros(ns)

for n, phi0 in enumerate(phi):
  for i in range(ns):
    hn[i] = hami(i, phi0, ka)

  energy[:,n] = np.linalg.eigvalsh(np.diag(hn))

np.savetxt('./config-landau-levels.txt', [phimin, phimax, nphi])
np.savetxt('./landau-levels.txt', energy, fmt='%1.4f')
