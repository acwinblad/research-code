#!/usr/bin/python3

import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import shapely.affinity as aff
import descartes as ds
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cbook
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pathlib

yp= np.sqrt(3)/2
plt.rcParams.update({'font.size': 13})

def build_hollow_triangle(_a, _nr, _width):
  # Innernr should always be at least 1 if we want a hollow triangle, but we can generalize to a full triangle for convenience.
  # The next available concentric triangle is always 3 less than the number of rows the triangle has.
  innernr = _nr-3*_width
  innerlen = _a * (innernr + 2)
  if(innernr < -1):
    innernr = innerlen = _width = 0
  outerlen = _a*(_nr-1)

  # Build triangle polygon, either full or hollow
  # Construct two triangles whose edges are the hollow triangles boundary (with tolerance)
  outertri = geo.Polygon([(-.5 * outerlen, 0), (0, yp * outerlen), (.5 * outerlen, 0)])
  if( _width == 0 ):
    innertri = geo.Polygon([(0,0),(0,0),(0,0)])
  elif( _width == 1 ):
    innertri = geo.Polygon([( -(outerlen - _a) / 2, yp * _a / 3), ( 0, yp * (3 * outerlen - 2 * _a) / 3), ( (outerlen - _a ) / 2, yp / 3) ])
  else:
    #innertri = geo.Polygon([(-.5 * innerlen, yp * (_width - 1 * _a)), (0, yp * (innerlen + _width - 1 * _a)), (.5 * innerlen, yp * (_width - 1 * _a))])
    innertri = geo.Polygon([(0.5*innerlen-_a/2, _width*_a*yp),
                            (0.5*innerlen-_a, (_width-1)*_a*yp),
                            (-0.5*innerlen+_a, (_width-1)*_a*yp),
                            (-0.5*innerlen+_a/2, _width*_a*yp),
                            (-_a/2, (_nr-2*_width)*_a*yp),
                            (_a/2, (_nr-2*_width)*_a*yp)])
  # Shift triangle downward to center it around its centroid
  outertri = aff.translate(outertri, xoff=0, yoff=-yp*outerlen/3)
  innertri = aff.translate(innertri, xoff=0, yoff=-yp*outerlen/3)
  # Take difference of the two triangles to get a hollow triangle
  hollowtri = outertri.symmetric_difference(innertri)
  return hollowtri, innertri

def create_directory_path(_vecPotType, _mu, _nr, _width):
  # Create figure file name. if mu is positive it'll have a 'p' and if negative it'll have 'n'
  filepath = './data/figures/%s-vector-potential/' % _vecPotType
  filepath += 'nr-%i/w-%i/' % (_nr, _width)
  absmu = abs(_mu)
  mustring = f"{absmu:1.4f}"
  mustring = mustring.replace('.','_')
  if(_mu>= 0):
    mustring = 'p' + mustring
  else:
    mustring = 'n' + mustring
  filepath2 = filepath + 'mu-%s/' % mustring
  pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
  pathlib.Path(filepath2).mkdir(parents=True, exist_ok=True)
  return filepath, filepath2

# This function is used to test if we are getting the triangular structure we want
def plot_hollow_triangle_lattice(_a, _nr, _hollowtri, _innertri, _filepath):
  n = _nr*(_nr+1)//2
  siteCoord = np.zeros((n,2))
  latticeCtr = 0
  for i in range(_nr):
    for j in range(i+1):
      siteCoord[latticeCtr,0] = _a*(j-i/2)
      siteCoord[latticeCtr,1] = (_nr-1-i)*_a*yp
      latticeCtr += 1

  # Shift all sites downward to center it around its centroid
  siteCoord[:,1] += -yp*_a*(_nr-1)/3
  fig = plt.figure(1, dpi=90)
  ax = fig.add_subplot(111)
  x,y = _hollowtri.exterior.xy
  x2,y2 = _innertri.exterior.xy
  patchhollow = ds.PolygonPatch(_hollowtri, alpha=0.25, color='blue')
  ax.add_patch(patchhollow)
  ax.plot(x,y, 'b')
  ax.plot(x2,y2, 'b')
  ax.plot(siteCoord[:,0],siteCoord[:,1], '.k')
  ax.set_aspect('equal')
  plt.savefig(_filepath + 'hollow-triangle-lattice.pdf')
  plt.close()
  plt.clf()
  plt.cla()

# This is to give us an idea of what the vector potential looks like on our triangular structure
def plot_vector_potential(_coords, _filepath):
  x = _coords[:,0]
  y = _coords[:,1]
  triang = mtri.Triangulation(x, y)
  plt.figure()
  plt.plot(x,y, '.', markersize=1.0, color='#15d4d4')
  #plt.ylim(-4,np.max(y)+2)
  plt.quiver(triang.x,triang.y, 0, triang.x/10, units='x',color='#d42e20', pivot='mid', width=0.15, scale=0.8)

  plt.savefig(_filepath + 'vector-potential-field.pdf')
  plt.close()
  plt.clf()
  plt.cla()

# Find lattice point coordinates inside the hollow triangle, used for nearest-neighbor
def hollow_triangle_coords(_a, _nr, _hollowtri):
  coords = np.empty((1,2))
  for i in range(_nr):
    for j in range(i+1):
      xi = _a*(j-i/2)
      yi = (_nr-1-i)*_a*yp - _a*yp*(_nr-1)/3
      if( _hollowtri.buffer(0.001*_a).intersects(geo.Point(xi,yi)) ):
        coords = np.append(coords,np.array([[xi, yi]]), axis=0)
  coords = np.delete(coords,0,0)
  return coords

# returns the distance between two points
def dist(_dx, _dy):
  return np.sqrt(_dx**2+_dy**2)

def calc_phase_factor(_dx, _dy):
  phase = np.arctan(_dy / _dx)
  if _dx<0:
    phase += np.pi
  return np.exp(-1.0j*phase)

def calc_phi(_a, _xj, _xl, _yj, _yl, _dx, _dy, _t, _vecPotFunc):
  #_t = _t % np.pi
  m = _dy / _dx
  b = _yl - m * _xl
  ct = np.cos(_t)
  st = np.sin(_t)
  rotdl = m * ct - st

  ## The Vector strength A will be treated as positive, some functions have the negative factor included in the integrand or phi
  if(_vecPotFunc == 'step-function'):
    ## Needs numerical integration
    xarr = np.linspace(_xj, _xl, 2001)
    yarr = m * xarr + b
    x1 = xarr * ct + yarr * st
    integrand = (1 - 2 * np.heaviside(x1, 0.5)) * rotdl
    phi = np.trapz(integrand,xarr)

  elif(_vecPotFunc == 'tanh'):
    ## Needs numerical integration
    alpha = 1000
    xarr = np.linspace(_xj, _xl, 4001)
    yarr = m * xarr + b
    x1 = xarr * ct + yarr * st
    integrand = -np.tanh(alpha * x1) * rotdl
    phi = np.trapz(integrand,xarr)

  elif(_vecPotFunc == 'linear'):
    ## Integral is done by hand
    phi = -rotdl * (0.5 * (ct + m * st) * (_xl**2 - _xj**2) + b * st * _dx)

  elif(_vecPotFunc == 'constant'):
    ## Integral is done by hand
    phi = rotdl * _dx

  return np.exp(1.0j * phi)

# Lists the nearest neighbor as an integer list
def nearest_neighbor_list(_a, _coords):
  n = np.size(_coords[:,0])
  outerlen = 2*np.max(_coords[:,0])
  nnlist = [ [] for i in range(n-1)]
  nnphaseFtr = [ [] for i in range(n-1)]
  nnphiParams = [ [] for i in range(n-1)]
  for j in range(n-1):
    for l in range(j+1,n):
      dx = _coords[l,0] - _coords[j,0]
      dy = _coords[l,1] - _coords[j,1]
      d = dist(dx,dy)
      if(np.abs(d-_a) < 1e-5*_a):
        nnlist[j].append(l)
        nnphaseFtr[j].append(calc_phase_factor(dx, dy))
        nnphiParams[j].append([dx, dy])

  return nnlist, nnphaseFtr, nnphiParams


def clone_width_one_interior_points(_a, _coords):
  x = _coords[:,0]
  y = _coords[:,1]
  ymin = np.min(y)
  n = np.size(x)
  length = 2*np.max(x)
  dups = np.zeros(n, dtype=bool)
  # create a triangle polygon with the corners cut off.
  interiorEdge = geo.Polygon([( -(length - _a) / 2, yp * _a),
                          ( -(length / 2 - _a), 0),
                          (length / 2 - _a, 0),
                          ( (length - _a) / 2, yp * _a),
                          ( _a / 2, (length - _a) * yp),
                          ( -_a / 2, (length - _a) * yp)])

  # find all the points that lie on this polygon's edge
  interiorEdge = aff.translate(interiorEdge, xoff=0, yoff=ymin)
  for i in range(n):
    dups[i] = interiorEdge.buffer(0.001*_a).intersects(geo.Point(x[i], y[i]))

  x2 = x[dups]
  y2 = y[dups]

  #since we are going to shift the clonedCoords we return dups bool array to duplicate eigenstate elements later
  y0 = np.where(y2 < ymin + 1e-5*_a)
  xNeg = np.where(np.logical_and(x2<0, y2>ymin+1e-5*_a))
  xPos = np.where(np.logical_and(x2>0, y2>ymin+1e-5*_a))
  shift = 0.10*_a
  y2[y0] += shift
  x2[xNeg] += yp*shift/2
  y2[xNeg] += -shift/2
  x2[xPos] += -yp*shift/2
  y2[xPos] += -shift/2
  clonedCoords = np.vstack([x2,y2]).T

  return clonedCoords, dups

def plot_hollow_triangle_wavefunction_circles(_a, _width, _nr, _coords, _tvals, _energy, _states0, _states1, _filepath):
  plt.rcParams.update({'font.size': 16})
  x = _coords[:,0]
  y = _coords[:,1]
  xmin = np.min(x)
  xmax = np.max(x)
  ymin = np.min(y)
  ymax = np.max(y)
  n = np.size(x)

  nl = _nr * _a
  padd = nl // 8
  paddins = 4
  inssize = 1.2

  vgs0 = _states0[0:n,:] + _states0[n:2*n,:]
  vgs1 = _states1[0:n,:] + _states1[n:2*n,:]
  wavefunction = vgs0+vgs1
  wfmax = np.max(wavefunction)

  vmin = 0
  vmax = wfmax

  figpreface = _filepath + 'GS'
  for j, angle in enumerate(_tvals):
    Tstring = f"{_tvals[j]:1.4f}"
    Tstring = Tstring.replace('.','_')
    Tstring = '-T-%s' % Tstring
    figname = figpreface + Tstring

    fig, ax = plt.subplots(dpi=300)

    ax.set_aspect('equal')
    #ax.set_xlabel('$x\ (a)$', fontsize=16)
    #ax.set_ylabel('$y\ (a)$', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title('$\epsilon$=%1.4e, $\eta$=%1.4e' % (_energy[j], _tvals[j]))
    plt.xlim(xmin-padd,xmax+padd)
    plt.ylim(ymin-padd,ymax+padd)

    #plt.triplot(triang, '.k', markersize=1.0, markeredgecolor='none')
    #vmax = np.max(wavefunction[:,j])
    im = plt.scatter(x,y, s=(301/(nl+2*padd))**2 * (20*wavefunction[:,j]+0.75)**2, c=wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax, alpha=0.9)
    #im = plt.scatter(x,y, s=(301/(_nr+1))**2, c=wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax)
    #im = plt.scatter(x,y+1.5, s=(201/(_nr+1))**2, c=wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    fig.colorbar(im, cax = cax, label = '$\|\Psi|^2$')
    plt.tight_layout()
    #plt.quiver(0, 0, np.cos(angle+np.pi/2), np.sin(angle+np.pi/2))

    axins = inset_axes(ax, width= inssize, height = inssize, loc=2)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_aspect('equal')
    axins.scatter(x,y, s = (73*inssize/(_width-_a + 1.5*paddins))**2 * (20*wavefunction[:,j]+0.75)**2, c = wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax, alpha=0.9)
    axins.set_xlim(xmin-_a*paddins/2, xmin + (_width - _a + _a*paddins))
    axins.set_ylim(ymin-_a*paddins/2, ymin + (_width - _a + _a*paddins))
    if(_width == 0):
      axins.set_xlim(xmin-_a*paddins/2, xmin + _a*paddins)
      axins.set_ylim(ymin-_a*paddins/2, ymin + _a*paddins)

    axins.set_xticklabels([])
    axins.set_yticklabels([])

    axins2 = inset_axes(ax, width= inssize, height = inssize, loc=1)
    axins2.set_xticks([])
    axins2.set_yticks([])
    axins2.set_aspect('equal')
    axins2.scatter(x,y, s = (73*inssize/(_width-_a + 1.5*paddins))**2 * (20*wavefunction[:,j]+0.75)**2, c = wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax, alpha=0.9)
    axins2.set_xlim(xmax - (_width - _a + _a*paddins), xmax + _a*paddins/2)
    axins2.set_ylim(ymin-_a*paddins/2, ymin + (_width - _a + _a*paddins) )
    if(_width == 0):
      axins2.set_xlim(xmax-_a*paddins/2, xmax + _a*paddins)
      axins2.set_ylim(ymin-_a*paddins/2, ymin + _a*paddins)

    axins2.set_xticklabels([])
    axins2.set_yticklabels([])

    plt.savefig(figname+'.pdf')
    plt.close()
    plt.clf()
    plt.cla()

def plot_quad_hollow_triangle_wavefunction_circles(_a, _width, _nr, _coords, _tvals, _energy, _states, _filepath):
  plt.rcParams.update({'font.size': 16})
  x = _coords[:,0]
  y = _coords[:,1]
  xmin = np.min(x)
  xmax = np.max(x)
  ymin = np.min(y)
  ymax = np.max(y)
  Dy = ymax-ymin

  xm = x + 2*xmax + _a
  xr = xm + 2*xmax + _a
  xt = x + _a + xmax

  xmax = np.max(xr)

  xtrip = np.concatenate((x,xm,xr,xt))
  ytrip = np.concatenate((y,y,y,y+Dy))

  n = np.size(x)
  nt = np.size(_tvals)

  nl = _nr * _a
  padd = nl // 8
  paddins = 4
  inssize = 1.2

  wavefunction = np.zeros((4*n, nt))
  for j in range(4):
    tmp0 = np.concatenate((_states[j, 0:n, :], _states[j,2*n:3*n,:], _states[j,4*n:5*n,:], _states[j,6*n:7*n,:]))
    tmp1 = np.concatenate((_states[j, n:2*n, :], _states[j,3*n:4*n,:], _states[j,5*n:6*n,:], _states[j, 7*n:8*n,:]))
    wavefunction += tmp0 + tmp1

  wfmax = np.max(wavefunction)

  vmin = 0
  vmax = wfmax

  figpreface = _filepath + 'GS'
  for j, angle in enumerate(_tvals):
    Tstring = f"{_tvals[j]:1.4f}"
    Tstring = Tstring.replace('.','_')
    Tstring = '-T-%s' % Tstring
    figname = figpreface + Tstring

    fig, ax = plt.subplots(dpi=300)

    ax.set_aspect('equal')
    #ax.set_xlabel('$x\ (a)$', fontsize=16)
    #ax.set_ylabel('$y\ (a)$', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title('$\epsilon$=%1.4e, $\eta$=%1.4e' % (_energy[j], _tvals[j]))
    plt.xlim(xmin-padd,xmax+padd)
    plt.ylim(ymin-padd,ymax+Dy+padd)

    #plt.triplot(triang, '.k', markersize=1.0, markeredgecolor='none')
    vmax = np.max(wavefunction[:,j])
    im = plt.scatter(xtrip,ytrip, s=(301/(1.5*nl+1+2*padd))**2 * (20*wavefunction[:,j]+0.75)**2, c=wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax, alpha=0.9)
    #im = plt.scatter(x,y, s=(301/(_nr+1))**2, c=wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax)
    #im = plt.scatter(x,y+1.5, s=(201/(_nr+1))**2, c=wavefunction[:,j], cmap='plasma', linewidths=0, vmin=vmin, vmax=vmax)
    #fig.colorbar(im, ax = ax, pad=0.010, label = '$\|\Psi|^2$')
    plt.tight_layout()
    plt.quiver(2*xmax/5 + 2*_a, 0, np.cos(angle+np.pi/2), np.sin(angle+np.pi/2))

    plt.savefig(figname+'.pdf')
    plt.close()
    plt.clf()
    plt.cla()

def plot_hollow_triangle_spectral_flow(_mu, _nr, _A0, _width, _nE, _avals, _eva, _filepath):
  plt.rcParams.update({'font.size': 16})
  #ymax = 1.01*np.max(_evb[:,:])
  ymax = 0.4
  ymin = -ymax
  xmax = np.max(_avals)
  xmin = np.min(_avals)

  plt.figure()
  plt.xlim(xmin,xmax)
  plt.ylim(ymin, ymax)
  plt.xlabel(r"$A$", fontsize=20)
  plt.xticks(np.linspace(xmin, np.int(xmax), np.int(xmax)+1))
  plt.ylabel('Energy (t)', fontsize=18)
  plt.locator_params(axis='y', nbins=5)
  plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))

  for i in range(2*_nE):
    plt.plot(_avals,_eva[i,:], 'C0')

  plt.tight_layout()
  plt.savefig(_filepath+'spectral-flow.pdf')
  plt.close()
  plt.clf()
  plt.cla()

def plot_hollow_triangle_rotation_spectral_flow(_mu, _nr, _A0, _width, _nE, _avals, _eva, _filepath):
  plt.rcParams.update({'font.size': 16})
  #ymax = 1.01*np.max(_evb[:,:])
  ymax = 0.2
  ymin = -ymax
  xmax = _avals[-1]
  xmin = _avals[0]

  plt.figure()
  #plt.xlim(xmin/xmax,xmax/xmax)
  plt.xlim(xmin,xmax)
  plt.ylim(ymin, ymax)
  #plt.xlabel(r"$\varphi$ ($\pi$)", fontsize=20)
  plt.xlabel(r"$\varphi$", fontsize=20)
  #plt.xticks(np.linspace(xmin,xmax, 5)/xmax)
  plt.xticks(np.linspace(xmin,xmax, 5))
  plt.ylabel('Energy (t)', fontsize=18)
  plt.locator_params(axis='y', nbins=5)
  plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))

  for i in range(2*_nE):
    plt.plot(_avals,_eva[i,:], 'C0')

  #plt.axvline(x=1/3, color='k', linestyle = "--")
  #plt.axvline(x=2/3, color='k', linestyle = "--")
  #plt.plot(0.0,0,"s",c='C1', markersize=10, clip_on=False, zorder=100)
  #plt.plot(1/6,0,"o",c='C1', markersize=10)
  #plt.plot(1/3,0,"D",c='C1', markersize=10)
  #plt.plot(2/3,0,"^",c='C1', markersize=10)

  plt.tight_layout()
  plt.savefig(_filepath+'spectral-flow.pdf')
  plt.close()
  plt.clf()
  plt.cla()

