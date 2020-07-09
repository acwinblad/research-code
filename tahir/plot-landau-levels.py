#/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

energy = np.loadtxt('./landau-levels.txt')
config = np.loadtxt('./config-landau-levels.txt')
phimin = config[0]
phimax = config[1]
nphi = int(config[2])
phi0 = np.linspace(phimin, phimax, nphi)

Emin = 3.99-0.001
Emax = 4.00001+0.001
nE = 250
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j], energy[:,i]<E[j+1]))[0]
    gE[i, j+1] = np.size(idx)

plt.figure(1000,figsize=(6,6))
ax = plt.gca()
plt.tick_params(
    axis='x',
    which='both',
    bottom='on',
    top='off',
    labelbottom='on')
plt.ylabel('$E$', fontsize=8)
plt.xlabel('$\phi_0$', fontsize=8)
plt.xlim(phimin, phimax)
plt.ylim(Emin, Emax)
ax.set_facecolor('black')
ax.set_aspect('auto')
ax.set(adjustable='box-forced', aspect='equal')
Extent = [0,phimax, Emin, Emax]
y_labels = ax.get_yticks()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
x_labels = ax.get_xticks()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0e'))

cmap='Blues'
plt.title('Landau levels')
plt.imshow(gE.transpose(), cmap=cmap, origin='lower', interpolation='bicubic', extent=Extent, aspect='auto')
plt.colorbar(fraction=0.05, pad=0.04)
plt.savefig('./fig-landau-levels-is.pdf')
plt.contourf(phi0, E, gE.transpose(), 30, cmap=cmap)
#plt.colorbar(fraction=0.046, pad=0.04)
plt.savefig('./fig-landau-levels-cf.pdf')


plt.close(1000)


