#!/usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

a = 1
w = 3
mui = -3.5
muf = -2.0

A0 = 0*np.pi
Af = 4*np.pi / (np.sqrt(3) *a)
Af = 1*np.pi

bg = np.loadtxt('./data/band-gap-constant-angle-w-%01i.txt' % (w) )

plt.rcParams.update({'font.size':13})

xmin = A0
xmax = Af
ymin = mui
ymax = muf
vmin = 0
vmax = np.max(bg)

plt.figure()
fig, ax = plt.subplots(1,1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.ylabel("$\mu$", fontsize=16)
plt.yticks(np.linspace(ymin, ymax, 2*np.int32(ymax-ymin)+1))
plt.yticks(np.linspace(ymin, ymax, 11))
plt.xlabel("$A$", fontsize=16)
nx = 6
ax.set_xticks(np.linspace(xmin,xmax,nx+1))
#plt.locator_params(axis='x', nbins=12)
plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))
plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))

#cax = ax.imshow(bpov, cmap='Accent', extent=[xmin,xmax,ymin,ymax], vmin=0, vmax=7)
cmap = plt.get_cmap('inferno')
#ticks = np.arange(1,17,2)*7/16
#cax = ax.imshow(bg, cmap=cmap, extent=[xmin,xmax,ymin,ymax], vmin=vmin, vmax=vmax)
cax = ax.imshow(bg, cmap=cmap, extent=[xmin,xmax,ymin,ymax], norm=colors.LogNorm())
cbar = fig.colorbar(cax, pad=0.01)
#cbar.ax.set_yticklabels(['[1,1,1]','[-1,1,1]','[-1,-1,1]','[1,-1,1]','[1,-1,-1]','[1,1,-1]','[-1,1,-1]','[-1,-1,-1]'])
cbar.set_label('$\mathcal{M}_{b,r,l}$', rotation=270, fontsize=18, labelpad=20)
ax.set_aspect('auto')
fig.tight_layout()
plt.savefig('./data/figures/band-gap-constant-angle-w-%01i.pdf' % (w) )
plt.close()
plt.clf()
plt.cla()



