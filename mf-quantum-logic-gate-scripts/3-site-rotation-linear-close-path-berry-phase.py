#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

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

p01 = 0
p12 = -pi/3
p20 = -pi/3
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
phi = np.array([[0, -pi/3, -pi/3], [-pi/3, -pi/3, 0], [-pi/3, 0, -pi/3], [0, -pi/3, -pi/3]])

tteng = np.zeros((3*(nphi-1),8))

for j in range(3):
  plist = np.linspace(phi[j,:], phi[j+1,:], nphi, endpoint=False)

  for l in range(nphi-1):
    hbdg = ( -t * ( np.exp(1.0j * plist[l+1,0]) * np.dot(np.conjugate(np.transpose(cop[0,:,:])), cop[1,:,:])
                   + np.exp(1.0j * plist[l+1,1]) * np.dot(np.conjugate(np.transpose(cop[1,:,:])), cop[2,:,:])
                   + np.exp(1.0j * plist[l+1,2]) * np.dot(np.conjugate(np.transpose(cop[2,:,:])), cop[0,:,:]) )
           + delta * ( np.exp(1.0j * th01) * np.dot(cop[0,:,:], cop[1,:,:])
                     + np.exp(1.0j * th12) * np.dot(cop[1,:,:], cop[2,:,:])
                     + np.exp(1.0j * th20) * np.dot(cop[2,:,:], cop[0,:,:]) ) )

    hbdg = hbdg + np.conjugate(np.transpose(hbdg))

    teng, tvec = np.linalg.eigh(hbdg)
    tteng[j*(nphi-1)+l,:] = teng

    tproj = np.outer(tvec[:,0],np.conjugate(tvec[:,0])) + np.outer(tvec[:,1],np.conjugate(tvec[:,1]))

    projgs = np.round(np.dot(tproj, projgs), decimals=12)

BerryPhase = np.round(np.dot(np.conjugate(np.transpose(gs)), projgs), decimals=8)
print(BerryPhase)
ArgBerryPhase = np.angle(BerryPhase)
print(ArgBerryPhase)
print( (ArgBerryPhase[0,0] - ArgBerryPhase[1,1]) / pi )

plt.figure()
for i in range(8):
  if(i<4):
    plt.plot(tteng[:,i]+0.00*i+1.25)
  else:
    plt.plot(tteng[:,i]+0.00*i-1.25)

plt.show()
plt.close()
