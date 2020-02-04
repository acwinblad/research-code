#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# Using the formulations given in B. Zhou, et al. PRL 101, 246807 (2008)
# We will be plotting an approximate eigenstate of a equilateral triangle
# This will not include the wavefunctions orthogonal variable, x, as we 
# want to see how the function looks like along the y-axis as L(x)->0.
np.set_printoptions(formatter={'float': lambda x: "{0:1.2e}".format(x)})

a = 364.5
b = -68.6
m = -10.
f = (a**2-2*m*b)/(2*b**2)
kx = 0.
e = a*kx

lp = np.sqrt( kx**2 + f + np.sqrt( f**2 - ( m**2 - e**2 ) / b**2 ) )
lm = np.sqrt( kx**2 + f - np.sqrt( f**2 - ( m**2 - e**2 ) / b**2 ) )

w = 0.5
h = np.sqrt(3/4)*200
hm = h*2
nx = 500
ny = nx
x = np.linspace(0, w-1e-15, nx)
y = np.zeros((ny, nx))  
tmpy = np.linspace(h, 0, ny)
for i in range(nx):
  hmax = h-hm*x[i]
  idx = np.where(tmpy <= hmax)
  y[idx,i] = tmpy[idx]

hmax = h-hm*x
cp = np.cosh(lp*hmax/2)
cm = np.cosh(lm*hmax/2)
sp = np.sinh(lp*hmax/2)
sm = np.sinh(lm*hmax/2)
tp = np.tanh(lp*hmax/2)
tm = np.tanh(lm*hmax/2)
cop = 1. / tp
com = 1. / tm

fp = np.divide( np.cosh(lp*(np.subtract(y,hmax/2))) , cp ) - np.divide( np.cosh(lm*(np.subtract(y,hmax/2))) , cm )
fm = np.divide( np.sinh(lp*(np.subtract(y,hmax/2))) , sp ) - np.divide( np.sinh(lm*(np.subtract(y,hmax/2))) , sm )

alp = e - m + b * ( kx**2 - lp**2 )
alm = e - m + b * ( kx**2 - lm**2 )
eta1 = ( b * ( lp**2 - lm**2 ) / a ) / ( lp * cop - lm * com )
eta2 = ( b * ( lp**2 - lm**2 ) / a ) / ( lp * tp - lm * tm )
gammap = ( b * ( lp**2 - lm**2 ) * kx * ( eta1 / eta2 ) ) / ( alm * lp * tp - alp * lm * tm )
gammam = - ( b * ( lp**2 - lm**2 ) * kx * ( eta1 / eta2 ) ) / ( alm * lp * tp - alp * lm * tm )

psi1 = fp + np.multiply(fm,gammap)
psi2 = eta1 * fp + np.multiply(np.multiply(fm,gammap),eta2)
psinorm = np.multiply(psi1.conj(),psi1) + np.multiply(psi2.conj(),psi2)
print(np.sum(psinorm[:,0]))
col_sums = psinorm.sum(axis=0)
psinorm /= np.sum(col_sums)*h/ny
print(np.sum(psinorm[:,0]))
psinorm[:,-1] = 0


plt.figure()
p = plt.imshow(psinorm)
plt.colorbar(p)
#plt.show()
plt.savefig('./fig-corner-state-approx.pdf')
plt.close()

