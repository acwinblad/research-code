#!/usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

a = 1
n = 3
mui = -5
muf = 3
Af = 4*np.pi / (np.sqrt(3) *a)

tp = np.loadtxt('./data/majorana-number-1pi6-n-%01i.txt' % (n))

plt.rcParams.update({'font.size':13})

xmin = 0
xmax = Af
ymin = mui
ymax = muf
vmin = np.min(tp)*1.2
vmax = np.max(tp)*1.2

plt.figure()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.xlabel("$A$", fontsize=16)
plt.xticks(np.linspace(xmin, np.int32(xmax), np.int32(xmax)+1))
plt.ylabel(r"$\mu$ (t)", fontsize=16)
plt.locator_params(axis='y', nbins=5)
plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.0f}"))

plt.imshow(tp, cmap='inferno_r', extent=[xmin,xmax,ymin,ymax], vmin=vmin, vmax=vmax)
plt.tight_layout()
plt.savefig('./data/figures/topological-phase-diagram-1pi6-n-%01i.pdf' % (n) )
plt.close()
plt.clf()
plt.cla()



