#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

#np.set_printoptions(threshold='nan')

runfile = 0
# property values 
Vx = 0.
Vy = 10.
Vz = 1000.
delta = 50.
t = 10.
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
k2 = -a*(x+np.sqrt(3)*y)/2.
k3 = a*(x-np.sqrt(3)*y)/2.
b1 = np.exp(-1.0j*np.pi/6.)
b2 = np.exp(1.0j*np.pi/6.)

# build the BdG Hamiltonian components
eps = -2.*t*(np.cos(k2) + np.cos(k3) + np.cos(k1)) -(mu-6*t)
#delP =2*delta*1j*(-np.exp(1j*4*np.pi/3.)*np.sin(k2) + np.exp(1j*5*np.pi/3.)*np.sin(k3) + np.sin(k1))
alpha1 = 2.*alpha*( -1.0j*np.sin(k1)+b1*np.sin(k2)+b2*np.sin(k3) )
alpha2 = np.conj(alpha1)
H_Z = np.matrix( [[Vz, Vx-1.0j*Vy, 0, 0],\
                  [Vx+1.0j*Vy, -Vz, 0, 0],\
                  [0, 0, -Vz, -Vx-1.0j*Vy],\
                  [0, 0, -Vx+1.0j*Vy, Vz]] )

# calculate the Energy spectrum
# this will be a 4x4 Hamiltonian, we will have to solve the eigenvalue problem at every coordinate point then categorize the energies correctly
E = np.zeros((4,np.size(x)))
idx = np.where(y==0)[0]
for i in range(np.size(x)):
  H_BdG = np.matrix( [[eps[i], alpha1[i], 0, delta],\
                      [alpha2[i], eps[i],-delta, 0],\
                      [0, -delta, -eps[i], alpha2[i]],\
                      [delta, 0, alpha1[i], -eps[i]]])
  H_BdG += H_Z
  energy, states = np.linalg.eigh(H_BdG)
  E[:,i] = np.real(energy)
  #E[:,i] = np.sort(np.real(energy))


band_gap = np.zeros(2)
band_gap[0] = np.max(E[1,:])
band_gap[1] = np.min(E[2,:])
#print(band_gap)
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
  plt.show()

