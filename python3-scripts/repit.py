#!/usr/bin/python

import os

os.system('rep ./configs/template-run-tbr runfile.rep -t tagfile.txt')
os.system('rep ./configs/template-tbr-lattice *.rep -t tagfile.txt')
os.system('rep ./configs/template-tbr-momentum *.rep -t tagfile.txt')
os.system('rep ./configs/template-plot-energy-lattice runfile.rep -t tagfile.txt')
