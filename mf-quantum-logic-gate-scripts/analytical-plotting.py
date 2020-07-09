#!/usr/bin/python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Zeeman field strengths
# vx isn't used but keeping here if we want to use it later, which will also mean we will have to change how we solve k later
#vx = 0.
vy = 0.
vz = 1000.

# Material values
delta = 25.0
th = 10.0
tr = 0.1*th
m = 1.0/(2.0*th)
mu = vz+6.0*th
energy = 0.0

n = 100
xm = 0.5
ym = np.sqrt(3)/2
x = np.linspace(-xm,xm,n)
y = np.linspace(0,ym,n)

# Our momentum values are j and k, kx and ky, respectively, we want to see how k changes when j changes since it's easier to compute k(j) than j(k)
# Number of j and k values
njk = 101
j = np.arange(0,njk,1)
# we have a standard quadratic equation, so we will define a=1, then b and c are
b = np.array(2*m*mu - 0.5*(m*tr*delta/vz)**2, dtype='complex')
c = np.array((m*delta)**2 * (vy**2 + 2*tr*vy*j/3) / vz + 8*m**2*tr*vy*energy*j/(3*vz) - (2*m*tr*vy*j/(3*vz))**2 + 4*m**2*(mu+energy)*(mu-energy), dtype='complex')

# The quadratic equation is for l^2 = j^2 / 9 + k^2 / 3, to simplify write l^2 as such, then write out k(j)
lsqp = -b + np.sqrt(b**2-c)
lsqm = -b - np.sqrt(b**2-c)
kp = np.sqrt(3*(lsqp-(j/3)**2))
km = np.sqrt(3*(lsqm-(j/3)**2))

#print(np.real(2*np.pi*1.0j*kp))
#print()
#print(np.real(-2*np.pi*1.0j*kp))
#print()
#print(np.real(2*np.pi*1.0j*km))
#print()
#print(np.real(-2*np.pi*1.0j*km))
X, Y = np.meshgrid(x,y[::-1])
print(X)
print(Y)

plt.figure()
plt.imshow(np.abs(np.exp(2*np.pi*1.0j*(j[0]*X+kp[0]*Y) ) - np.exp(2*np.pi*1.0j*(j[0]*X-kp[0]*Y) ) )**2 )
#plt.plot(j,1.0j*(kp-km), 'g+')
#plt.plot(j,1.0j*kp, 'b_')
#plt.plot(j,1.0j*km, '_')
#plt.plot(j,1.0j*-kp,'g_')
#plt.plot(j,1.0j*-km,'r_')
plt.show()
plt.close()
