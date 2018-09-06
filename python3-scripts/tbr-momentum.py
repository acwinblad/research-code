#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

#np.set_printoptions(threshold='nan')

runfile = '%runfile%'
#property values
Vx = %vx%
Vy = %vy%
Vz = %vz%
delta = %delta% 
t = %t%
mu = Vz+%mufctr%*t
alpha = %alphafctr%*t
#runfile = 'test'
##property values
#Vx = 0.
#Vy = 25.
#Vz = 1000.
#delta = 25.
#t = 10.
#mu = Vz+6*t
#alpha = 0.25*t
a = 1.

yslice = True
contour_flag = False
plot_flag = True

# dimensions for the graph
nr = %nrm%
#nr = 500
d = 2*np.pi/(a*np.sqrt(3))
ds = d/nr
x = np.zeros(3*nr*(nr+1)+1)
y = np.zeros(3*nr*(nr+1)+1)

# build up the hexagonal domain with triangles
latticeCtr = 0
for i in range(nr+1):
  for j in range(nr+i+1):
    x[latticeCtr] = -np.sqrt(3)*d/2.+np.sqrt(3)*i*ds/2.
    y[latticeCtr] = d/2.+i*ds/2.-j*ds
    latticeCtr+=1
for i in range(nr):
  for j in range(2*nr-i):
    x[latticeCtr] = np.sqrt(3)*(i+1)*ds/2.
    y[latticeCtr] = d-(i+1)*ds/2.-j*ds
    latticeCtr+=1
triang = mtri.Triangulation(x,y)

# build the momentum axis to be utilized
k1 = a*x
k2 = a*(x-np.sqrt(3)*y)/2.
k3 = -a*(x+np.sqrt(3)*y)/2.
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
#filename = './data/%s-tbr' % runfile
filename = './%s-tbr' % runfile
np.savetxt(filename+'-energy-band-gaps.txt', band_gap)

# plot density of states
for i in range(2):
  nE = 150
  Eumax = np.max(E[2*i+1,:])
  Eumin = np.min(E[2*i+1,:])
  Elmax = np.max(E[2*i,:])
  Elmin = np.min(E[2*i,:])
  dEu = np.abs(Eumax-Eumin)/(nE-1)
  dEl = np.abs(Elmax-Elmin)/(nE-1)
  Eaxis = np.array([Elmin + j*dEl for j in range(nE)])
  Eaxis = np.append(Eaxis, np.array([Eumin+j*dEu for j in range(nE)]))
  dos = np.zeros(np.size(Eaxis))
  for j in range(nE-1):
    idx = np.where(np.logical_and(E[2*i,:]<Eaxis[j+1],E[2*i,:]>Eaxis[j]))[0]
    dos[j] = np.size(idx)
  for j in range(nE-1):
    idx = np.where(np.logical_and(E[2*i+1,:]<Eaxis[nE+j+1],E[2*i+1,:]>Eaxis[nE+j]))[0]
    dos[nE+j+1] = np.size(idx)

  #plt.xlabel('$\epsilon (t)$', fontsize=12)
  #plt.ylabel('$g(\epsilon)/g_{max}$', fontsize=12)
  #plt.xlim(Elmin,Eumax)
  #plt.plot(Eaxis,dos)
  #plt.show()
  #plt.savefig('../../data/fig-dos-momentum.pdf')


#plot energy band gap
fig = plt.figure()
if yslice == True:
  idx = np.where(y==0)[0]
  plt.grid(True)
  #plt.plot(x[idx],E[0,idx])
  plt.plot(x[idx],E[1,idx])
  plt.plot(x[idx],E[2,idx])
  #plt.plot(x[idx],E[3,idx])
elif contour_flag == True:
  triang = mtri.Triangulation(x,y)
  plt.axis('equal')
  plt.grid(True)
  plt.tricontour(triang,E[2,:], 25, cmap='plasma')
else:
  skipcols = 2
  ax = fig.add_subplot(111, projection = '3d')
  triang = mtri.Triangulation(x[::skipcols],y[::skipcols])
  # plot energy in k-space
  ax.xaxis.label.set_size(12)
  ax.yaxis.label.set_size(12)
  ax.zaxis.label.set_size(12)
  ax.set_xlim(-np.pi/a,np.pi/a)
  ax.set_ylim(-d,d)
  ax.set_xlabel(r'$k_x\ (1/a)$')
  ax.set_ylabel(r'$k_y\ (1/a)$')
  ax.set_zlabel(r'$\epsilon\ (t)$')
  ax.view_init(elev=10., azim=-60)
  ax.view_init(elev=0., azim=-90)
  ax.plot_trisurf(triang, E[0,::skipcols], cmap='viridis')
  ax.plot_trisurf(triang, E[1,::skipcols], cmap='viridis')
  ax.plot_trisurf(triang, E[2,::skipcols], cmap='viridis')
  ax.plot_trisurf(triang, E[3,::skipcols], cmap='viridis')
if plot_flag == True:
  plt.show()

#fig.savefig('fig-'+filename+'.pdf')
