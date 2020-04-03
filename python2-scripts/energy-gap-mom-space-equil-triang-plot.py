#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

# property values 
delta = 1.0+0j
#t = 10*np.abs(delta)
t = 1
mu = 6*t
a = 1.

# dimensions for the graph
d = 2*np.pi/(a*np.sqrt(3))
nr = 250
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

# calculate the Energy_+/- 
eps = -2*t*(np.cos(a*x/2.+np.sqrt(3)*a*y/2.) + np.cos(a*x/2.-np.sqrt(3)*a*y/2.) + np.cos(a*x)) -(mu-6*t)
delP =2*delta*1j*(-np.exp(1j*4*np.pi/3.)*np.sin(a*x/2.+np.sqrt(3)*a*y/2.) + np.exp(1j*5*np.pi/3.)*np.sin(a*x/2.-np.sqrt(3)*a*y/2.) + np.sin(a*x))
Ep = np.sqrt(eps**2+np.real(np.conj(delP)*delP))
Em = -np.sqrt(eps**2+np.real(np.conj(delP)*delP))

grad = np.gradient(Ep)
idx = np.where(np.abs(grad)<1e-5)[0]
Epmin = np.min(Ep[idx])
np.savetxt('../../data/energy-band-gap-bounds.txt', np.column_stack([-Epmin,Epmin]))

# plot density of states
nE = 75
dE = (np.max(Ep)-Epmin)/(nE-1)
nE0 = int(Epmin/dE)
E = np.array([i*dE for i in range(nE0)])
E = np.append(E,np.array([i*dE+Epmin for i in range(nE)]))
gE = np.zeros(nE+nE0)
for i in range(nE0+nE-1):
  idx = np.where(np.logical_and(Ep<E[i+1],Ep>E[i]))[0]
  gE[i+1] = np.size(idx)

E = np.hstack([-np.flipud(E),E])
gE = np.hstack([np.flipud(gE),gE])/np.max(gE)

plt.xlabel('$\epsilon (t)$', fontsize=12)
plt.ylabel('$g(\epsilon)/g_{max}$', fontsize=12)
plt.xlim(np.min(E),np.max(E))
plt.plot(E,gE)
#plt.show()
plt.savefig('../../data/fig-dos-momentum.pdf')

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
triang = mtri.Triangulation(x[::5],y[::5])
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
ax.plot_trisurf(triang, Ep[::5], cmap='viridis')
ax.plot_trisurf(triang, Em[::5], cmap='viridis')
fig.savefig('../../data/fig-energy-band-gap-momentum-space.pdf')

