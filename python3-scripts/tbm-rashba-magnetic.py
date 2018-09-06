#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

#np.set_printoptions(threshold='nan')

# property values 
delta = 1.0
t = 10.
mu = 6.*t
a = 1.
alpha = 0.5*t
#alpha = 0
Vx = 0.
Vy = 0.0
Vz = 100.

yslice = True
contour_flag = False
plot_flag = False

# dimensions for the graph
nr = 500
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
k2 = -a*(x+np.sqrt(3)*y)/2.
k3 = a*(x-np.sqrt(3)*y)/2.
b1 = -1.0j
b2 = np.exp(1.0j*np.pi/6.)
b3 = np.exp(-1.0j*np.pi/6.)

# build the BdG Hamiltonian components
eps = -2.*t*(np.cos(k2) + np.cos(k3) + np.cos(k1)) -(mu-6*t)
#delP =2*delta*1j*(-np.exp(1j*4*np.pi/3.)*np.sin(k2) + np.exp(1j*5*np.pi/3.)*np.sin(k3) + np.sin(k1))
alpha1 = -2.*alpha*( b1*np.sin(k1)+b2*np.sin(k2)+b3*np.sin(k3) )
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
                      [np.conj(alpha1[i]), eps[i],-delta, 0],\
                      [0, -delta, -eps[i], np.conj(alpha1[i])],\
                      [delta, 0, alpha1[i], -eps[i]]])
  H_BdG += H_Z
  energy, states = np.linalg.eigh(H_BdG)
  #E[:,i] = np.real(energy)
  E[:,i] = np.sort(np.real(energy))


band_gap = np.zeros(4)
for i in range(4):
  band_gap[i] = np.max(E[i,:]*(-1)**i)*(-1)**i
#print(band_gap)
#np.savetxt('../../data/tbm-momentum-rashba-in-plane-magnetic-energy-band-gaps.py', band_gap)

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
  plt.plot(x[idx],E[0,idx])
  plt.plot(x[idx],E[1,idx])
  plt.plot(x[idx],E[2,idx])
  plt.plot(x[idx],E[3,idx])
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
  #ax.plot_trisurf(triang, E[2,::skipcols], cmap='viridis')
  #ax.plot_trisurf(triang, E[3,::skipcols], cmap='viridis')
if plot_flag == True:
  plt.show()

# solve in lattice space

nr = 30
n = nr*(nr+1)/2

bdg = np.zeros((4*n,4*n),dtype='complex')
h_z = np.zeros((4*n,4*n),dtype='complex')

# create the equilateral triangle lattice mesh
siteCoord = np.zeros((n,2))
latticeCtr = 0
for i in range(nr):
  for j in range(i+1):
    siteCoord[latticeCtr,0] = a*(j-i/2.)
    siteCoord[latticeCtr,1] = -i*a*np.sqrt(3)/2.
    latticeCtr+=1

# fill in the bdg Hamiltonian without the pairing potential then add it's H.C.
for i in range(n):
  for j in range(n-i):
    dx = siteCoord[i+j,0]-siteCoord[i,0]
    dy = siteCoord[i+j,1]-siteCoord[i,1]
    d = np.sqrt(dx**2+dy**2)
    if d<1e-5:
      # this is the current lattice site
      bdg[i,i]         =  (-mu+6*t)/2.
      bdg[i+n,i+n]     = -(-mu+6*t)/2.
      bdg[i+2*n,i+2*n] =  (-mu+6*t)/2.
      bdg[i+3*n,i+3*n] = -(-mu+6*t)/2.
      bdg[i,i+3*n]   =  delta
      bdg[i+2*n,i+n] = -delta
      h_z[i,i+2*n]   =  (Vx-1.0j*Vy)
      h_z[i+3*n,i+n] = -(Vx-1.0j*Vy)
      h_z[i+2*n,i]   =  (Vx+1.0j*Vy)
      h_z[i+n,i+3*n] = -(Vx+1.0j*Vy)
      h_z[i,i]         =  Vz
      h_z[i+n,i+n]     = -Vz
      h_z[i+2*n,i+2*n] = -Vz
      h_z[i+3*n,i+3*n] =  Vz
    elif np.abs(d-a) < 1e-5:
      # this is the nearest neighbor
      phaseAngle = np.arctan(dy/dx)
      bdg[i+j,i]         = -t
      bdg[i+n,i+j+n]     =  t
      bdg[i+j+2*n,i+2*n] = -t
      bdg[i+3*n,i+j+3*n] =  t
      # we have three angles to consider, arctan will return 0.0, 1.047, and -1.047
      # it is easier to write a >,=,< than compare the phase angles
      if phaseAngle == 0.0:
        bdg[i+j,i+2*n]   = -alpha
        bdg[i+3*n,i+j+n] =  alpha
        bdg[i+j+2*n,i]   =  alpha
        bdg[i+n,i+j+3*n] = -alpha
      elif phaseAngle > 0:
        bdg[i+j,i+2*n]   = alpha*np.exp(1.0j* (phaseAngle/2.-np.pi))
        bdg[i+3*n,i+j+n] = alpha*np.exp(1.0j* (phaseAngle/2.-np.pi))
        bdg[i+j+2*n,i]   = alpha*np.exp(-1.0j*(phaseAngle/2.+np.pi))
        bdg[i+n,i+j+3*n] = alpha*np.exp(-1.0j*(phaseAngle/2.+np.pi))
      # else the phase angle is negative
      elif phaseAngle < 0:
        bdg[i+j,i+2*n]   = alpha*np.exp(-1.0j*(phaseAngle/2.+np.pi))
        bdg[i+3*n,i+j+n] = alpha*np.exp(-1.0j*(phaseAngle/2.+np.pi))
        bdg[i+j+2*n,i]   = alpha*np.exp(1.0j* (phaseAngle/2.-np.pi))
        bdg[i+n,i+j+3*n] = alpha*np.exp(1.0j* (phaseAngle/2.-np.pi))

bdg = bdg + np.conj(np.transpose(bdg)) + h_z
energy, states = np.linalg.eigh(bdg)

idx = energy.argsort()[::-1]
energy = np.real(energy[idx])
states = states[:,idx]
states = np.real(np.multiply(states,np.conj(states)))
np.savetxt('../../data/tbm-lattice-rashba-in-plane-magnetic-bdg.py', bdg, fmt='%1.1f')
np.savetxt('../../data/tbm-lattice-rashba-in-plane-magnetic-energy.py', energy, fmt='%1.8f')
np.savetxt('../../data/tbm-lattice-rashba-in-plane-magnetic-states.py', states, fmt='%1.32f')
np.savetxt('../../data/tbm-lattice-rashba-in-plane-magnetic-coordinates.py', siteCoord, fmt='%1.32f')

xaxis = np.array([-1,1])

plt.figure(figsize=(2,4))
plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')
plt.ylabel('$\epsilon (t)$', fontsize=12)
plt.xlim(-1.2,1.2)
for i in range(4*n):
  plt.plot(xaxis, [energy[i],energy[i]], 'b', linewidth=1)
plt.tight_layout()
plt.show()
#xaxis = np.linspace(0,4*n,4*n)
#plt.xlim(0,4*n)
#plt.grid("True")
#plt.plot(xaxis,energy)
#plt.plot(xaxis,np.abs(energy))
#plt.show()

energyGap = np.loadtxt('../../data/tbm-momentum-rashba-in-plane-magnetic-energy-band-gaps.py')

nE = 75
nEG = 15 # needs to be odd

Eumax = np.max(energy[energy>=0])
Eumin = np.min(energy[energy>=0])
Elmax = np.max(energy[energy<0])
Elmin = np.min(energy[energy<0])

dEumax = np.abs(Eumax-energyGap[3])/(nE-1)
dEugap = np.abs(band_gap[3]-band_gap[2])/(nEG-1)
dEumin = np.abs(band_gap[2]-Eumin)/(nE-1) 
dElmax = np.abs(Elmax-energyGap[1])/(nE-1)
dElgap = np.abs(band_gap[1]-band_gap[0])/(nEG-1)
dElmin = np.abs(band_gap[0]-Elmin)/(nE-1)

Eaxis = np.array([Elmin+j*dElmin for j in range(nE)])
Eaxis = np.append(Eaxis, np.array([band_gap[0]+j*dElgap for j in range(nEG)]))
Eaxis = np.append(Eaxis, np.array([band_gap[1]+j*dElmax for j in range(nE)]))
Eaxis = np.append(Eaxis, np.array([Eumin+j*dEumin for j in range(nE)]))
Eaxis = np.append(Eaxis, np.array([band_gap[2]+j*dEugap for j in range(nEG)]))
Eaxis = np.append(Eaxis, np.array([band_gap[3]+j*dEumax for j in range(nE)]))

nes = np.size(np.where(np.logical_and(energy<band_gap[1],energy>band_gap[0]))[0])
print(nes)
dos = np.zeros(np.size(Eaxis))
for j in range(2*nE+nEG-1):
  idx = np.where(np.logical_and(energy<Eaxis[j+1],energy>Eaxis[j]))[0]
  dos[j] = np.size(idx)
for j in range(2*nE+nEG-1):
  idx = np.where(np.logical_and(energy<Eaxis[j+1+2*nE+nEG],energy>Eaxis[j+2*nE+nEG]))[0]
  dos[j+2*nE+nEG+1] = np.size(idx)

plt.grid("True")
plt.ylim(0,1)
plt.xlim(band_gap[0]-2*dElmin,band_gap[1]+2*dElmax)
plt.plot([band_gap[0],band_gap[0]],[0,1], 'red')
plt.plot([band_gap[1],band_gap[1]],[0,1], 'red')
plt.plot([band_gap[2],band_gap[2]],[0,1], 'red')
plt.plot([band_gap[3],band_gap[3]],[0,1], 'red')
plt.plot(Eaxis,dos/np.max(dos), 'black')
plt.show()
