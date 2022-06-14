#!/usr/bin/python3

import numpy as np

t = 1
#delta = t*1.1
delta = t
mu = 6.0

a = 1
nr = 5
n = nr*(nr+1)//2
B = -8.*np.pi / (3*np.sqrt(3)*a**2 * (2*nr-3) )

def checkBC(_s1,_s2,_a,_nr):
  if(_s1[1] == np.sqrt(3)*_s1[0] and _s2[1] == np.sqrt(3)*_s2[0]):
    y1flag = True
  else:
    y1flag = False

  if(_s1[1] == -np.sqrt(3)*_s1[0] and _s2[1] == -np.sqrt(3)*_s2[0]):
    y2flag = True
  else:
    y2flag = False

  if(_s1[1] == np.sqrt(3)*_a*(_nr-1)/2 and _s2[1] == np.sqrt(3)*_a*(_nr-1)/2):
    y3flag = True
  else:
    y3flag = False

  if(y1flag or y2flag or y3flag):
    return True
  else:
    return False



bdg = np.zeros((2*n,2*n),dtype='complex')

siteCoord = np.zeros((n,2))
latticeCtr = 0
for i in range(nr):
  for j in range(i+1):
    siteCoord[latticeCtr,0] = a*(j-i/2)
    siteCoord[latticeCtr,1] = -i*a*np.sqrt(3)/2
    latticeCtr += 1

for i in range(n):
  for j in range(n-i):
    if(checkBC(siteCoord[i,:], siteCoord[i+j,:], a, nr)):
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
print(eng[n-30:n])

np.savetxt('./data/kitaev-triangle-coord.txt', siteCoord, fmt='%1.32f')
np.savetxt('./data/kitaev-triangle-energy.txt', eng, fmt='%1.8e')
np.savetxt('./data/kitaev-triangle-states.txt', vec, fmt='%1.8e')
