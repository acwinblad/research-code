#!/usr/bin/python3

import hollow_triangle_module as htm
import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
import matplotlib.pyplot as plt
import imageio
import os
from pdf2image import convert_from_path



# Define parameters
t = 1
delta = t
mu = 1.5*t
nr = 100
width = 1

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
vecPotFunc = 'constant'
#vecPotFunc = 'tanh'

latticePlotPath, filepath = htm.create_directory_path('rotation-' + vecPotFunc, mu, nr, width)

frames = []
for file_name in sorted(os.listdir(filepath)):
  if (file_name.startswith('GS') and file_name.endswith('pdf')):
    file_path = os.path.join(filepath, file_name)
    image = convert_from_path(file_path, 300)
    file_path_re = file_path.replace('pdf','png')
    image[0].save(file_path_re)
    frames.append(imageio.imread(file_path_re))
    os.remove(file_path_re)

imageio.mimsave('./rotation-' + vecPotFunc + '-vector-potential.gif', frames, fps = 5, loop=    1)


