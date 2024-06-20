#!/usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

a = 1
w = 3
mui = -5
muf = 3
Af = 4*np.pi / (np.sqrt(3) *a)
#Af = 4*np.pi

tp = np.loadtxt('./data/majorana-number-1pi3-w-%01i.txt' % (w))

plt.rcParams.update({'font.size':13})

xmin = 0
xmax = Af
ymin = mui
ymax = muf
vmin = np.min(tp)*1.2
vmax = np.max(tp)*1.2

plt.figure()
fig, ax = plt.subplots(1,1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.xlabel("$A$", fontsize=16)
plt.xticks(np.linspace(xmin, np.int32(xmax), np.int32(xmax)+1))
plt.ylabel(r"$\mu$ (t)", fontsize=16)
plt.locator_params(axis='y', nbins=5)
plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.0f}"))

#cax = ax.imshow(tp, cmap='inferno_r', extent=[xmin,xmax,ymin,ymax], vmin=vmin, vmax=vmax)
cmap = plt.get_cmap('inferno',4)
cax = ax.imshow(tp, cmap=cmap, extent=[xmin,xmax,ymin,ymax], vmin=0, vmax=3)
ticks = np.arange(1,9,2)*3/8
cbar = fig.colorbar(cax, ticks=ticks, pad=0.01)
cbar.set_ticklabels(['[1,1]','[-1,1]','[1,-1]','[-1,-1]'])
#cbar.set_label('$(\mathcal{M}_b,\mathcal{M}_t)$', rotation=270, fontsize=16)
cbar.set_label('$\mathcal{M}_{b,t}$', rotation=270, fontsize=18, labelpad=15)
ax.set_aspect('auto')
fig.tight_layout()
plt.savefig('./data/figures/topological-phase-diagram-1pi3-w-%01i.pdf' % (w) )
plt.close()
plt.clf()
plt.cla()



