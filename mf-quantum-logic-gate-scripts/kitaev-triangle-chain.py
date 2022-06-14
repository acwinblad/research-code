#!/usr/bin/python3

import numpy as np

t = 1
#delta = t*1.1
delta = t
mu = 6.0

a = 1
nr = 100
#n = nr*(nr+1)//2
n = 3*nr-2
B = 0*8.*np.pi / (3*np.sqrt(3)*a**2 * (2*nr-3) )

bdg = np.zeros((2*n,2*n),dtype='complex')

siteCoord = np.zeros((n,2))
latticeCtr = 0
siteCoord[latticeCtr,0] = 0
siteCoord[latticeCtr,1] = 0
for i in range(nr-2):
  for j in range(2):
    siteCoord[latticeCtr,0] = a*(i+1)/2 * (-1)**(j+1)
    siteCoord[latticeCtr,1] = -(i+1)*a*np.sqrt(3)/2
    latticeCtr += 1
for i in range(nr):
  siteCoord[latticeCtr,0] = a*(-(nr-1)/2+i)
  siteCoord[latticeCtr,1] = -a*np.sqrt(3)/2 * (nr-1)
  latticeCtr += 1

for i in range(n):
  for j in range(n-i):
    dx = siteCoord[i+j,0]-siteCoord[i,0]
    dy = siteCoord[i+j,1]-siteCoord[i,1]
    d = np.sqrt(dx**2 + dy**2)

    if d<1e-5:
      bdg[i,i] = -mu+6*t
      bdg[i+n,i+n] = -bdg[i,i]
    elif np.abs(d-a)<1e-5:
      phaseAngle = np.arctan(dy/dx)

      if dx<0:
        phaseAngle += np.pi
      phi = -(B/2) * (dy/dx) * (siteCoord[i+j,0]**2 - siteCoord[i,0]**2)
      bdg[i+j,i] =     -t*np.exp(1.0j*phi)
      bdg[i,i+j] =    -t*np.exp(-1.0j*phi)
      bdg[i+j+n,i+n] = t*np.exp(-1.0j*phi)
      bdg[i+n,i+j+n] =  t*np.exp(1.0j*phi)
      bdg[i+j,i+n] = delta*np.exp(1.0j*phaseAngle)
      bdg[i,i+j+n] = -delta*np.exp(1.0j*phaseAngle)
      bdg[i+n,i+j] = delta*np.exp(-1.0j*phaseAngle)
      bdg[i+j+n,i] = -delta*np.exp(-1.0j*phaseAngle)

eng, vec = np.linalg.eigh(bdg)
#vec = vec[0:n,0:n]
#vec = np.real( np.matmul(vec.conj().T,vec) )
vec = np.real(np.multiply(vec,np.conj(vec)))
print(eng[n-5:n+5])

np.savetxt('./data/kitaev-triangle-chain-coord.txt', siteCoord, fmt='%1.32f')
np.savetxt('./data/kitaev-triangle-chain-energy.txt', eng, fmt='%1.8e')
np.savetxt('./data/kitaev-triangle-chain-states.txt', vec, fmt='%1.8e')
