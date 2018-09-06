#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

#np.set_printoptions(threshold='nan')

runfile = 'pm-basis'
# property values 
Vx = 0.
Vy = 10.
Vz = 1000.
delta = 50.
t = 10.
m = 1./(2.*t)
mu = Vz+3*t
alpha = 0.5*t

#runfile = sys.argv[1]
#Vx = np.float(sys.argv[2])
#Vy = np.float(sys.argv[3])
#Vz = np.float(sys.argv[4])
#delta = np.float(sys.argv[5])
#t = np.float(sys.argv[6])
#mu = Vz+np.float(sys.argv[7])*t
#alpha = np.float(sys.argv[8])*t
a = 1.

plot_flag = True

# dimensions for the graph
nr = 500
#nr = np.int(sys.argv[9])
d = 2*np.pi/(a*np.sqrt(3))
ds = d/nr
x = np.zeros(2*nr+1)
y = np.zeros(2*nr+1)

# build up the hexagonal domain with triangles
latticeCtr = 0
for i in range(2*nr+1):
  x[latticeCtr] = -np.sqrt(3)*d/2.+np.sqrt(3)*i*ds/2.
  y[latticeCtr] = 0
  latticeCtr+=1
#triang = mtri.Triangulation(x,y)

# build the momentum axis to be utilized
k1 = a*x

# build the BdG Hamiltonian components
gammap = Vy/alpha-k1
gammam = Vy/alpha+k1
etap = np.sqrt(gammap**2)
etam = np.sqrt(gammam**2)
epprime = np.sqrt(Vz**2+(alpha*etap)**2)
emprime = np.sqrt(Vz**2+(alpha*etam)**2)
Aup = alpha*etap*np.sqrt(1./(2*epprime*(epprime-Vz)))
Aum = alpha*etam*np.sqrt(1./(2*emprime*(emprime-Vz)))
Adp = Aup*(Vz-epprime)/(alpha*etap)
Adm = Aum*(Vz-emprime)/(alpha*etam)
fpp = Aup*Adm
fpm = Aum*Adp
fsp = Aup*Aum-Adp*Adm*gammap*gammam/(etap*etam)
fsm = np.conj(fsp)
dpp = delta*fpp*1.0j*gammam/ etam
dpm = delta*fpm*1.0j*gammap/ etap
dmp = delta*fpm*1.0j*gammap/ etap
dmm = delta*fpp*1.0j*gammam/ etam
dpmp = delta*fsp
dpmm = delta*fsm
epp = k1**2/(2*m) - mu + epprime
epm = k1**2/(2*m) - mu + emprime
emp = k1**2/(2*m) - mu - epprime
emm = k1**2/(2*m) - mu - emprime

# calculate the Energy spectrum
# this will be a 4x4 Hamiltonian, we will have to solve the eigenvalue problem at every coordinate point then categorize the energies correctly
E = np.zeros((4,np.size(x)))
idx = np.where(y==0)[0]
for i in range(np.size(x)):
  H_BdG = np.matrix( [[ epp[i], dpp[i]-dpm[i], 0, dpmp[i]],\
                      [ np.conj(dpp[i]-dpm[i]), -epm[i], -np.conj(dpmm[i]), 0],\
                      [ 0, -dpmm[i], emp[i], dmp[i]-dmm[i]],\
                      [ np.conj(dpmp[i]), 0, np.conj(dmp[i]-dmm[i]), -emm[i]]])
  energy, states = np.linalg.eigh(H_BdG)
  #E[:,i] = np.real(energy)
  E[:,i] = np.sort(np.real(energy))


band_gap = np.zeros(2)
band_gap[0] = np.max(E[1,:])
band_gap[1] = np.min(E[2,:])
print(band_gap)
#idx = np.where(E[1,:] == band_gap[0])[0]
#print(E[2,idx])
#idx = np.where(E[2,:] == band_gap[1])[0]
#print(E[1,idx])
#filename = '../../data/tbm-lattice-rashba-in-plane-magnetic-%s' % runfile
#np.savetxt(filename+'-energy-band-gaps.txt', band_gap)
if plot_flag==True:
  plt.figure()
  plt.tight_layout()
  plt.grid("True")
  plt.plot(k1, E[1,:], 'black')
  plt.plot(k1, E[2,:], 'blue')
  #plt.plot(k1, Ep, 'green')
  #plt.plot(k1, Em, 'red')
  plt.show()

