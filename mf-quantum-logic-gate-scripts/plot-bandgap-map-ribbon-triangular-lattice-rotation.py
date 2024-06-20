#!/usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

a = 1
w = 3
mu = -2.50
varphi0 = 0
varphif = np.pi

A0 = 0*np.pi
Af = 4*np.pi / (np.sqrt(3) *a)
Af = 1*np.pi

absmu = abs(mu)
mustring = f"{absmu:1.4f}"
mustring = mustring.replace('.','_')
if(mu >= 0):
  mustring = 'p' + mustring
else:
  mustring = 'n' + mustring


bg = np.loadtxt('./data/band-gap-rotation-w-%01i-mu-%s.txt' % (w, mustring) )
nt = np.size(bg[0,:])

ntr = 1*nt//3
ntl = 2*nt//3

bgr = np.hstack([bg[:,ntr:], bg[:,0:ntr]])
bgl = np.hstack([bg[:,ntl:], bg[:,0:ntl]])
bgb = bg
bgmin = np.minimum(bgb,np.minimum(bgr,bgl))
#bgmin = (bgb+bgr+bgl)/3
#bgmin = np.power(np.square(bgb)+np.square(bgr)+np.square(bgl),0.5)

plt.rcParams.update({'font.size':13})

xmin = varphi0
xmax = varphif
ymin = A0
ymax = Af
vmin = 0
vmax = np.max(bg)

plt.figure()
fig, ax = plt.subplots(1,1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.ylabel("$A$", fontsize=16)
plt.yticks(np.linspace(ymin, np.round(2*ymax)/2, 4*np.int32(ymax-ymin)+1))
#plt.yticks(np.linspace(ymin, ymax, 11))
plt.xlabel(r"$\varphi$", fontsize=16)
nx = 6
ax.set_xticks(np.linspace(xmin,xmax,nx+1))
#plt.locator_params(axis='x', nbins=12)
plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))
plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))

#cax = ax.imshow(bpov, cmap='Accent', extent=[xmin,xmax,ymin,ymax], vmin=0, vmax=7)
cmap = plt.get_cmap('inferno')
#ticks = np.arange(1,17,2)*7/16
#cax = ax.imshow(bg, cmap=cmap, extent=[xmin,xmax,ymin,ymax], vmin=vmin, vmax=vmax)
cax = ax.imshow(bgmin, cmap=cmap, extent=[xmin,xmax,ymin,ymax], norm=colors.LogNorm())
cbar = fig.colorbar(cax, pad=0.01)
#cbar.ax.set_yticklabels(['[1,1,1]','[-1,1,1]','[-1,-1,1]','[1,-1,1]','[1,-1,-1]','[1,1,-1]','[-1,1,-1]','[-1,-1,-1]'])
cbar.set_label('Band gap $(t)$', rotation=270, fontsize=18, labelpad=20)
ax.set_aspect('auto')
fig.tight_layout()
plt.savefig('./data/figures/band-gap-rotation-w-%01i-mu-%s.pdf' % (w, mustring) )
plt.close()
plt.clf()
plt.cla()



