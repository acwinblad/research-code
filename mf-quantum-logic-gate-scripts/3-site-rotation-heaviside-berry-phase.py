#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
sq3 = np.sqrt(3)
A = 2*pi/(3*sq3)

m01 = 0
m12 = -sq3
m20 = sq3

b01 = -sq3/6
b12 = sq3/3
b20 = sq3/3


def phi01(_t):
  x = np.linspace(-0.5,0.5,5001)
  y = m01 * x + b01
  x1 = x * np.cos(_t) + y * np.sin(_t)
  rotdl = m01 * np.cos(_t) - np.sin(_t)
  integrand = (1 - 2 * np.heaviside(x1, 0.0)) * rotdl
  integral = A * np.trapz(integrand,x)
  return integral

def phi12(_t):
  x = np.linspace(0.5,0.0,5001)
  y = m12 * x + b12
  x1 = x * np.cos(_t) + y * np.sin(_t)
  rotdl = m12 * np.cos(_t) - np.sin(_t)
  integrand = (1 - 2 * np.heaviside(x1, 1.0)) * rotdl
  integral = A * np.trapz(integrand,x)
  return integral

def phi20(_t):
  x = np.linspace(0.0,-0.5,5001)
  y = m20 * x + b20
  x1 = x * np.cos(_t) + y * np.sin(_t)
  rotdl = m20 * np.cos(_t) - np.sin(_t)
  integrand = (1 - 2 * np.heaviside(x1, 0.0)) * rotdl
  integral = A * np.trapz(integrand,x)
  return integral

# Setup operator in Fock space with the following basis
# |0,0,0>
# |1,0,0>, |0,1,0>, |0,0,1>
# |0,1,1>, |1,0,1>, |1,1,0>
# |1,1,1>
# = |0>, |1>, ... , |7>

cop = np.zeros((3,8,8))
cop[0,0,1] = cop[0,2,6] = cop[0,3,5] = cop[0,4,7] = 1
cop[1,0,2] = cop[1,3,4] = 1
cop[1,1,6] = cop[1,5,7] = -1
cop[2,0,3] = cop[2,6,7] = 1
cop[2,1,5] = cop[2,2,4] = -1

# Setup Majorana fermion notation

ma = np.zeros((3,8,8), dtype = "complex")
mb = np.zeros((3,8,8), dtype = "complex")
ma[0,:,:] = cop[0,:,:] + np.transpose(cop[0,:,:])
ma[1,:,:] = cop[1,:,:] + np.transpose(cop[1,:,:])
ma[2,:,:] = cop[2,:,:] + np.transpose(cop[2,:,:])
mb[0,:,:] = -1.0j*(cop[0,:,:] - np.transpose(cop[0,:,:]))
mb[1,:,:] = -1.0j*(cop[1,:,:] - np.transpose(cop[1,:,:]))
mb[2,:,:] = -1.0j*(cop[2,:,:] - np.transpose(cop[2,:,:]))

# fermion annihilation operator for the initial corner mode

fgs = np.real((ma[0,:,:] + 1.0j * mb[1,:,:])) / 2

# fermion number operator for the initial corner mode

nmop = np.dot(np.conjugate(np.transpose(fgs)), fgs)

# initialize constants and initial states

t = 1
delta = t

p01 = phi01(0)
p12 = phi12(0)
p20 = phi20(0)
#print(phi01(0*pi/3),phi12(0*pi/3),phi20(0*pi/3))
#print(phi01(1*pi/3),phi12(1*pi/3),phi20(1*pi/3))
#print(phi01(2*pi/3),phi12(2*pi/3),phi20(2*pi/3))
#print(phi01(3*pi/3),phi12(3*pi/3),phi20(3*pi/3))
th01 = 0
th12 = 2*pi/3
th20 = -2*pi/3

# Setup initial Hamiltonian

hbdg0 = ( -t * ( np.exp(1.0j * p01) * np.dot(np.conjugate(np.transpose(cop[0,:,:])), cop[1,:,:])
               + np.exp(1.0j * p12) * np.dot(np.conjugate(np.transpose(cop[1,:,:])), cop[2,:,:])
               + np.exp(1.0j * p20) * np.dot(np.conjugate(np.transpose(cop[2,:,:])), cop[0,:,:]) )
     + delta * ( np.exp(1.0j * th01) * np.dot(cop[0,:,:], cop[1,:,:])
               + np.exp(1.0j * th12) * np.dot(cop[1,:,:], cop[2,:,:])
               + np.exp(1.0j * th20) * np.dot(cop[2,:,:], cop[0,:,:]) ) )

hbdg0 += np.conjugate(np.transpose(hbdg0))
eng, vec = np.linalg.eigh(hbdg0)

# Unitary GS operator
ug = vec[:,0:2]

nmopproj = np.real(np.dot(np.dot(np.conjugate(np.transpose(ug)), nmop),ug))

neng, nvec = np.linalg.eigh(nmopproj)

gs = np.round(np.dot(ug,nvec), decimals=12)

projgs = gs

nphi = 10000
phimax = pi
phi = np.linspace(0,phimax,nphi, endpoint=False)
tteng = np.zeros((nphi-1,8))

for j in range(nphi-1):
  hbdg = ( -t * ( np.exp(1.0j * phi01(phi[j+1])) * np.dot(np.conjugate(np.transpose(cop[0,:,:])), cop[1,:,:])
                + np.exp(1.0j * phi12(phi[j+1])) * np.dot(np.conjugate(np.transpose(cop[1,:,:])), cop[2,:,:])
                + np.exp(1.0j * phi20(phi[j+1])) * np.dot(np.conjugate(np.transpose(cop[2,:,:])), cop[0,:,:]) )
         + delta * ( np.exp(1.0j * th01) * np.dot(cop[0,:,:], cop[1,:,:])
                   + np.exp(1.0j * th12) * np.dot(cop[1,:,:], cop[2,:,:])
                   + np.exp(1.0j * th20) * np.dot(cop[2,:,:], cop[0,:,:]) ) )

  hbdg = hbdg + np.conjugate(np.transpose(hbdg))

  teng, tvec = np.linalg.eigh(hbdg)
  tteng[j,:] = teng

  tproj = np.outer(tvec[:,0],np.conjugate(tvec[:,0])) + np.outer(tvec[:,1],np.conjugate(tvec[:,1]))

  projgs = np.round(np.dot(tproj, projgs), decimals=12)

BerryPhase = np.round(np.dot(np.conjugate(np.transpose(gs)), projgs), decimals=8)
print(BerryPhase)
ArgBerryPhase = np.angle(BerryPhase)
print(ArgBerryPhase)
print( (ArgBerryPhase[0,0] - ArgBerryPhase[1,1]) / pi )

plt.figure()
for i in range(8):
  if( i < 4):
    plt.plot(tteng[:,i]+1.25)
  else:
    plt.plot(tteng[:,i]-1.25)


plt.show()
plt.close()
