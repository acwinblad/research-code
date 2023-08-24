#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import imageio

# Define parameters
t = 1
delta = t
mu = 0.0*t
a = 1
nr = 2
n = nr*(nr+1)//2
sqrt3 = np.sqrt(3)
A = 2*np.pi/(3*sqrt3*a)

bdg = np.zeros((2*n,2*n), dtype='complex')

def gj(_X,_Y,_xj, _yj, sig):
  return np.exp( ( (_X - _xj)**2 + (_Y + _yj)**2 ) / (-2 * sig**2) )

# chemical potential and order parameter will not change so we only need to initialize it once
bdg[0:n, 0:n] = -mu*np.eye(n)
bdg[n:2*n, n:2*n] = -bdg[0:n,0:n]

bdg[4,0] = -delta
bdg[3,1] = delta
bdg[5,0] = -delta*np.exp(-1.0j*np.pi/3)
bdg[3,2] = -bdg[5,0]
bdg[5,1] = -delta*np.exp(-1.0j*2*np.pi/3)
bdg[4,2] = -bdg[5,1]

dt = 30
dt += 1

p12 = np.linspace(0,-np.pi/3,dt)
p12 = np.append(p12, np.linspace(-np.pi/3,-np.pi/3,dt))
p12 = np.append(p12, np.linspace(-np.pi/3,0,dt))

p13 = np.linspace(-np.pi/3,0,dt)
p13 = np.append(p13, np.linspace(0,-np.pi/3,dt))
p13 = np.append(p13, np.linspace(-np.pi/3,-np.pi/3,dt))

p23 = np.linspace(-np.pi/3,-np.pi/3,dt)
p23 = np.append(p23, np.linspace(-np.pi/3,0,dt))
p23 = np.append(p23, np.linspace(0,-np.pi/3,dt))

eba0 = np.zeros((3,3*dt))
eba1 = np.zeros((3,3*dt))
vba0 = np.zeros((3,3*dt))
vba1 = np.zeros((3,3*dt))

for i, (pp12, pp13, pp23) in enumerate(zip(p12, p13, p23)):
  # replace the hopping parameter with the rotated phase
  bdg[1,0] = -t*np.exp(1.0j*pp12)
  bdg[4,3] = -np.conjugate(bdg[1,0])
  bdg[2,0] = -t*np.exp(-1.0j*pp13)
  bdg[5,3] = -np.conjugate(bdg[2,0])
  bdg[2,1] = -t*np.exp(1.0j*pp23)
  bdg[5,4] = -np.conjugate(bdg[2,1])

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(0.5*bdg)
  vec = np.real(np.multiply(vec, np.conj(vec)))

  eba0[:,i] = eng[0:n]
  vba0[:,i] = vec[0:n,n-1]+vec[n:2*n,n-1]
  eba1[:,i] = eng[n:]
  vba1[:,i] = vec[0:n,n]+vec[n:2*n,n]


fig, ax = plt.subplots(dpi=300)

plt.xlim(0,3*dt-1)
plt.ylabel('Energy (t)', fontsize=16)
plt.xlabel('')
ax.set_xticks([0, dt, 2*dt, 3*dt-1])
ax.set_xticklabels([r'$\phi_1$', r'$\phi_2$', r'$\phi_3$', r'$\phi_1$'])
ax.tick_params(axis='x', which='major', labelsize=16)

plt.plot(eba0[-1,:], c = 'C0')
plt.plot(eba0[1,:] , c = 'C0')
plt.plot(eba0[0,:] , c = 'C0')
plt.plot(eba1[-1,:], c = 'C0')
plt.plot(eba1[1,:] , c = 'C0')
plt.plot(eba1[0,:] , c = 'C0')

plt.axvline(x=dt, c = 'k', linestyle = "--")
plt.axvline(x=2*dt, c = 'k', linestyle = "--")

plt.plot(0   , 0, "s", c='C1', markersize=10, clip_on=False, zorder=100)
plt.plot(dt/2, 0, "o", c='C1', markersize=10)
plt.plot(dt  , 0, "D", c='C1', markersize=10)
plt.plot(2*dt, 0, "^", c='C1', markersize=10)

plt.tight_layout()
#plt.show()
plt.savefig('./3eigval.pdf')
plt.close()

#print(eba1)
#print(eba2)
#print(vba1)
#print(vba2)

wfmax = np.max(vba0[:,:]+vba1[:,:])
#wfmax = np.max(vba0[:,:])
wf = np.zeros((3,4))
shapesx = np.array([0, dt//2, dt, 2*dt])
shapes = ['s','o','D','^']
for i, shape in enumerate(shapesx):
  wf[:,i] = vba0[:,shape] + vba1[:,shape]


# Create a 2x2 plot of the wavefunctions
fig, ax = plt.subplots(2,2)
plt.rcParams.update({'font.size': 13})
vmin = 0
vmax = wfmax
cworder = [0,1,3,2]

for i, axes in enumerate(ax.flat):
  axes.set_xticks([])
  axes.set_yticks([])
  x = np.linspace(-a,a,1001)
  y = np.linspace(-a+a*np.sqrt(3)/6,a,1001)
  X,Y = np.meshgrid(x,y)
  Y -= a*np.sqrt(3)/6
  extent = [np.min(x), np.max(x), np.min(y), np.max(y)]

  Z =  wf[0,cworder[i]] * gj(X,Y,-a/2,-sqrt3*a/6, 0.1*a)
  Z += wf[1,cworder[i]] * gj(X,Y,a/2,-sqrt3*a/6, 0.1*a)
  Z += wf[2,cworder[i]] * gj(X,Y,0,sqrt3*a/3, 0.1*a)

  im = axes.imshow(Z, extent=extent, vmin = vmin, vmax = vmax, cmap = 'plasma')
  axes.plot(-a*0.80, a*0.80, marker = shapes[cworder[i]], c = 'C1', markersize=5, clip_on=False, zorder=100)

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
fig.subplots_adjust(wspace=-0.4)
fig.subplots_adjust(hspace=0.1)
#fig.colorbar(im, ax=ax.ravel().tolist(), label = '$\|\Psi|^2$', orientation = "horizontal", pad=0.05)
fig.colorbar(im, ax=ax, label = '$\|\Psi|^2$', location = "bottom", pad=0.05, shrink=0.72)

#plt.show()
plt.savefig('./3eigvec.pdf')
plt.close()


def create_frame(_i):
  fig = plt.figure(figsize=(6,6))
  # plot wavefunction here with plt functions
  x = np.linspace(-a,a,1001)
  y = np.linspace(-a,a,1001)
  X,Y = np.meshgrid(x,y)
  extent = [np.min(x), np.max(x), np.min(y), np.max(y)]

  Z =  (vba1[0,_i] - 1*vba0[0,_i])*gj(X,Y,-a/2,-sqrt3*a/6, 0.1*a)
  Z += (vba1[1,_i] - 1*vba0[1,_i])*gj(X,Y,a/2,-sqrt3*a/6, 0.1*a)
  Z += (vba1[2,_i] - 1*vba0[2,_i])*gj(X,Y,0,sqrt3*a/3, 0.1*a)
  plt.xlim([-a,a])
  plt.xlabel('x (a)', fontsize = 16)
  plt.ylim(-a*(1-sqrt3/6),a)
  plt.ylabel('y (a)', fontsize = 16)

  plt.imshow(Z, extent=extent)

  plt.savefig(f'./img/lcpr-img_{_i}.png', transparent = False, facecolor = 'white' )
  plt.close()

#for i, t in enumerate(p12):
#  create_frame(i)

frames = []
for i, t in enumerate(p12):
  image = imageio.imread(f'./img/lcpr-img_{i}.png')
  frames.append(image)

imageio.mimsave('./linear-closed-path-rotation-2vecdiff.gif', frames, fps = 10, loop=1)
