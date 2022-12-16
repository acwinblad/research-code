#!/usr/bin/python3

import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cbook
import pathlib

yp= np.sqrt(3)/2

def build_hollow_triangle(_a, _nr, _width):
  # Innernr should always be at least 1 if we want a hollow triangle, but we can generalize to a full t    riangle for convenience.
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
                            (-_a/2, (_nr-_width-3)*_a*yp),
                            (_a/2, (_nr-_width-3)*_a*yp)])
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
      yi = (_nr-1-i)*_a*yp
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

def calc_phi_factor(_a, _nnx, _x, _dx, _dy, _vecPotFunc):
  if(_vecPotFunc == 'step-function'):
    if( np.abs(_dy) >= 1e-5*_a):
      if(np.abs(_x) >= 1e-5*_a):
        phi = -yp*_a * np.sign(_x)
      else:
        phi = -yp*_a * np.sign(_nnx)
    else:
      phi = 0
  elif(_vecPotFunc == "linear"):
    phi = -0.5 * (_dy/_dx) * (_nnx**2 - _x**2)
  else:
    phi = 0

  return np.exp(1.0j*phi)

# Lists the nearest neighbor as an integer list
def nearest_neighbor_list(_a, _coords, _vecPotFunc):
  n = np.size(_coords[:,0])
  nnlist = [ [] for i in range(n-1)]
  nnphaseFtr = [ [] for i in range(n-1)]
  nnphiFtr = [ [] for i in range(n-1)]
  for i in range(n-1):
    for j in range(i+1,n):
      dx = _coords[j,0] - _coords[i,0]
      dy = _coords[j,1] - _coords[i,1]
      d = dist(dx,dy)
      if(np.abs(d-_a) < 1e-5*_a):
        nnlist[i].append(j)
        nnphaseFtr[i].append(calc_phase_factor(dx, dy))
        nnphiFtr[i].append(calc_phi_factor(_a, _coords[j,0], _coords[i,0], dx, dy, _vecPotFunc))

  return nnlist, nnphaseFtr, nnphiFtr


def clone_width_one_interior_points(_a, _coords):
  x = _coords[:,0]
  y = _coords[:,1]
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
  for i in range(n):
    dups[i] = interiorEdge.buffer(0.001*_a).intersects(geo.Point(x[i], y[i]))

  x2 = x[dups]
  y2 = y[dups]

  #since we are going to shift the clonedCoords we return dups bool array to duplicate eigenstate elements later
  y0 = np.where(abs(y2) < 1e-5*_a)
  xNeg = np.where(np.logical_and(x2<0, y2>0))
  xPos = np.where(np.logical_and(x2>0, y2>0))
  shift = 0.08*_a
  y2[y0] += shift
  x2[xNeg] += yp*shift/2
  y2[xNeg] += -shift/2
  x2[xPos] += -yp*shift/2
  y2[xPos] += -shift/2
  clonedCoords = np.vstack([x2,y2]).T

  return clonedCoords, dups

def shifted_innertri(_a, _nr):
  length = _a*(_nr-1)
  shift = _a * 0.08
  shift2 = yp * shift / 2
  shift3 = shift / 2
  interiorEdge = geo.Polygon([( -(length - _a) / 2 + shift2, yp * _a - shift3),
                              ( -(length / 2 - _a), shift),
                              (length / 2 - _a, shift),
                              ( (length - _a) / 2 - shift2, yp * _a - shift3),
                              ( _a / 2 - shift2, (length - _a) * yp - shift3),
                              ( -_a / 2 + shift2, (length - _a) * yp - shift3)])

  return interiorEdge

def create_centroids_mask(_a, _coords, _innertri):
  x = _coords[:,0]
  y = _coords[:,1]
  n = np.size(x)
  triang = mtri.Triangulation(x, y)
  centroids = np.zeros(n)
  centroids = np.vstack((x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)))
  mask = np.zeros(np.size(triang.triangles[:,0]), dtype=bool)

  for i, val in enumerate(mask):
    mask[i] = _innertri.buffer(0.020*_a).intersects(geo.Point(centroids[0,i], centroids[1,i]))
  triang.set_mask(mask)
  return triang

def plot_hollow_triangle_wavefunction(_a, _width, _innertri, _coords, _triang, _nE, _bval, _energy, _states, _filepath):
  x = _coords[:,0]
  y = _coords[:,1]
  n = np.size(x)
  triang = mtri.Triangulation(x, y)
  centroids = np.zeros(n)
  centroids = np.vstack((x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)))
  #centri = mtri.Triangulation(centroids[0,:], centroids[1,:])
  mask = np.zeros(np.size(triang.triangles[:,0]), dtype=bool)

  for i, val in enumerate(mask):
    mask[i] = _innertri.buffer(0.020*_a).intersects(geo.Point(centroids[0,i], centroids[1,i]))
  triang.set_mask(mask)
  idx = np.arange(-_nE, _nE)

  for i in idx:
    if(i <= 0):
      Estring = 'Ep' + f"{_nE+i:02d}"
    else:
      Estring = 'En' + f"{_nE-i-1:02d}"
    Bstring = f"{_bval}"
    absb = abs(_bval)
    Bstring = f"{absb:1.4f}"
    Bstring = Bstring.replace('.','_')
    Bstring = '-B-%s' % Bstring
    figname = _filepath + Estring + Bstring + '.pdf'

    fig, ax = plt.subplots()
    #plt.subplots_adjust(bottom=-0.01)

    #vmx = np.max(_states[0:n,i])
    #v = np.linspace(0,vmx,19)
    #clb = fig.colorbar(format='%1.4f')
    #clb.ax.set_ylabel('$\|\Psi\|^2$' , rotation=0, labelpad=15, fontsize=12)

    ax.set_aspect('equal')
    ax.set_xlabel('$x\ (a)$', fontsize=12)
    ax.set_ylabel('$y\ (a)$', fontsize=12)
    ax.set_title('$\epsilon$=%1.4e, $B$=%1.4e' % (_energy[i], _bval))
    im = ax.tricontourf(triang, _states[0:n,i]+_states[n:2*n,i], cmap='Blues')
    ax.tricontourf(triang, _states[0:n,i]+_states[n:2*n,i], cmap='Blues')
    ax.triplot(triang, '.k', markersize=1)
    #ax.triplot(centri, '.r', markersize=1)
    fig.colorbar(im, ax = ax, pad=0.010, label = '$\|\Psi\|^2$')

    axins = ax.inset_axes([0.02,0.70, 0.30, 0.30])
    axins.set_aspect('equal')
    axins.tricontourf(triang, _states[0:n,i]+_states[n:2*n,i], cmap='Blues')
    axins.triplot(triang, '.k', markersize=1)
    #axins.triplot(centri, '.r', markersize=1)
    axins.set_xlim(np.min(x),np.min(x)+_width+_a+_a/2)
    if(_width > 1):
      axins.set_ylim(0, np.sqrt(3)/2 * (_width + _a))
    elif(_width == 0):
      axins.set_xlim(np.min(x),np.min(x)+_a)
      axins.set_ylim(0, np.sqrt(3)/2 * 0.75*_a)
    else:
      axins.set_ylim(0, np.sqrt(3)/2 * _width * 1.25)

    axins.set_xticklabels([])
    axins.set_yticklabels([])

    axins2 = ax.inset_axes([0.68,0.70, 0.30, 0.30])
    axins2.set_aspect('equal')
    axins2.tricontourf(triang, _states[0:n,i]+_states[n:2*n,i], cmap='Blues')
    axins2.triplot(triang, '.k', markersize=1)
    #axins2.triplot(centri, '.r', markersize=1)
    axins2.set_xlim(np.max(x)-_width-_a-_a/2,np.max(x))
    if(_width > 1):
      axins2.set_ylim(0, np.sqrt(3)/2 * (_width + _a))
    elif(_width == 0):
      axins2.set_xlim(np.max(x)-_a,np.max(x))
      axins2.set_ylim(0, np.sqrt(3)/2 * 0.75*_a)
    else:
      axins2.set_ylim(0, np.sqrt(3)/2 * _width * 1.25)
    axins2.set_xticklabels([])
    axins2.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
    plt.clf()
    plt.cla()

def plot_hollow_triangle_spectral_flow(_mu, _nr, _B0, _width, _Bmax, _nE, _bvals, _evb, _filepath):
  ymin = np.min(_evb[0:_nE,:])
  ymax = -ymin
  plt.figure()
  plt.tight_layout()
  plt.grid(True)
  plt.xlim(0,_Bmax)
  #plt.xticks([round(-Bmax/4 * i, 2) for i in range(5)])
  plt.ylim(ymin, ymax)
  plt.xlabel('$B$', fontsize=12)
  plt.ylabel('Energy (t)', fontsize=12)
  if(_width!=0):
    plt.title('$n_r$ = %i, $w_{edge}$ = %i, $\mu$ = %1.4f' % (_nr, _width, _mu))
  else:
    plt.title('$n_r$ = %i, $\mu$ = %1.4f' % (_nr, _mu))
  plt.axvline(x=_B0, color='black', ls='--')
  for i in range(2*_nE):
    plt.plot(_bvals,_evb[i,:], 'b')
  #plt.show()
  plt.savefig(_filepath+'spectral-flow.pdf')
  plt.close()
  plt.clf()
  plt.cla()

