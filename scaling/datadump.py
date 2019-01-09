#! /usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib

import sys
import plotting as myplt


fname = "/home/alexeedm/extern/daint/scratch/datadump/raw.txt"

raw = pd.read_csv(fname, sep='\s+', header=None)
data = raw.values
nodes, compute, total, dump = [ data[:, i] for i in range(4) ]

myplt.set_pgf_backend()
fig = plt.figure(0)
ax = plt.gca()

plt.plot(nodes, compute, ':', label= 'Computation', linewidth=1.5, zorder=3, color='black', alpha=1.0)
plt.plot(nodes, dump, '--', label= 'Data dump', linewidth=1.5, zorder=3, color='C1')
plt.plot(nodes, total, '-', label= 'Total', linewidth=2.5, zorder=2, color='C2')
        
nodes = [ x**3 for x in [1,2,3,4,5,6] ]
strnodes = [ '$'+str(x)+'^3$' for x in [1,2,3,4,5,6] ]
ax.set_xscale("log", nonposx='clip')
#ax.set_yscale()
#ax.set_yscale("log", nonposy='clip')

#ax.set_yticks(nodes)
#ax.set_yticklabels(np.array(nodes, dtype=int))

ax.set_xticks(nodes)
ax.set_xticklabels(strnodes)

ax.set_xlabel("Nodes")
ax.set_ylabel("Time per step, \si{\milli\second}")
plt.legend()#frameon=False)

myplt.set_font_sizes(ax)
myplt.make_grid(ax)

myplt.save_figure(fig, 3, 'datadump.pdf')
#plt.show()
