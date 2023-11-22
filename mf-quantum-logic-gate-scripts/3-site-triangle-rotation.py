#!/usr/bin/python3

import hollow_triangle_module as htm
import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
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

def p12(_t):
  x = np.linspace(-0.5*a,0.5*a,1001)
  y = np.tanh(1000 * (x * np.cos(_t) - sqrt3 * a * np.sin(_t) / 6))
  return -a * A * np.trapz(y,x) * np.sin(_t)

def p13(_t):
  x = np.linspace(-0.5*a,0,1001)
  y = np.tanh(1000 * (x * (np.cos(_t) + sqrt3 * np.sin(_t)) + sqrt3 * a * np.sin(_t) / 3))
  return a * A * np.trapz(y,x) * (sqrt3 * np.cos(_t) - np.sin(_t))

def p23(_t):
  x = np.linspace(0.5*a,0,1001)
  y = np.tanh(1000 * (x * (np.cos(_t) - sqrt3 * np.sin(_t)) + sqrt3 * a * np.sin(_t) / 3))
  return -a * A * np.trapz(y,x) * (sqrt3 * np.cos(_t) + np.sin(_t))

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

dt = 180
dt += 1
angle = np.linspace(0,np.pi,dt)
eba0 = np.zeros((3,dt))
eba1 = np.zeros((3,dt))
vba0 = np.zeros((3,dt))
vba1 = np.zeros((3,dt))

for i, rot in enumerate(angle):
  # replace the hopping parameter with the rotated phase
  bdg[1,0] = -t*np.exp(-1.0j*p12(rot))
  bdg[4,3] = -np.conjugate(bdg[1,0])
  bdg[2,0] = -t*np.exp(-1.0j*p13(rot))
  bdg[5,3] = -np.conjugate(bdg[2,0])
  bdg[2,1] = -t*np.exp(-1.0j*p23(rot))
  bdg[5,4] = -np.conjugate(bdg[2,1])

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(bdg)
  vec = np.real(np.multiply(vec, np.conj(vec)))

  eba0[:,i] = eng[0:n]
  vba0[:,i] = vec[0:n,n-1]+vec[n:2*n,n-1]
  eba1[:,i] = eng[n:]
  vba1[:,i] = vec[0:n,n]+vec[n:2*n,n]

plt.figure()
#plt.tight_layout()
#plt.plot(angle,vba0[0,:])
#plt.plot(angle,vba0[1,:])
#plt.plot(angle,vba0[2,:])
#plt.plot(angle,eba0[-1,:])
#plt.plot(angle,eba0[1,:])
#plt.plot(angle,vba1[0,:]+0.51)
#plt.plot(angle,vba1[1,:]+0.51)
#plt.plot(angle,vba1[2,:]+0.51)
plt.close()
for i in range(n-1):
  plt.plot(angle,eba0[i+1,:])
  plt.plot(angle,eba1[i,:])
plt.show()
plt.close()

#print(eba1)
#print(eba2)
#print(vba1)
#print(vba2)

def create_frame(_i, _t):
  fig = plt.figure(figsize=(6,6))
  # plot wavefunction here with plt functions
  x = np.linspace(-a,a,1001)
  y = np.linspace(-a,a,1001)
  X,Y = np.meshgrid(x,y)
  extent = [np.min(x), np.max(x), np.min(y), np.max(y)]

  Z = vba0[0,_i]*gj(X,Y,-a/2,-sqrt3*a/6, 0.1*a)
  Z += vba0[1,_i]*gj(X,Y,a/2,-sqrt3*a/6, 0.1*a)
  Z += vba0[2,_i]*gj(X,Y,0,sqrt3*a/3, 0.1*a)
  plt.xlim([-a,a])
  plt.xlabel('x (a)', fontsize = 16)
  plt.ylim(-a*(1-sqrt3/6),a)
  plt.ylabel('y (a)', fontsize = 16)

  plt.imshow(Z, extent=extent)

  plt.savefig(f'./img/img_{_t}.png', transparent = False, facecolor = 'white' )
  plt.close()

#for i,t in enumerate(angle):
#  create_frame(i,t)

frames = []
for t in angle:
  image = imageio.imread(f'./img/img_{t}.png')
  frames.append(image)

imageio.mimsave('./example.gif', frames, fps = 10, loop=1)
