#! /usr/bin/env python

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib

import sys
import plotting as myplt

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs = "+")
parser.add_argument('--ref', type=int, default=1)
parser.add_argument('--out', type=str, default='guui')
parser.add_argument('--fontSize', type=int, default=22)

blood = True

if blood:
    args = parser.parse_args(['--files',
                              'data/fine_min_weak_periodic_blood_64.txt',
                              'data/fine_min_weak_periodic_blood_96.txt'])
else:
    args = parser.parse_args(['--files',
                              'data/fine_min_weak_periodic_poiseuille_64.txt',
                              'data/fine_min_weak_periodic_poiseuille_96.txt'])


if args.out == "gui":
    myplt.set_qt_backend()
else:
    myplt.set_pgf_backend()
    
fig = plt.figure(0)
ax = fig.add_subplot(1, 1, 1)

def getDomain(fname):
    tmp = fname.split("_")[-1]
    return tmp.split(".")[0]

nodes=[]


i=1
mstyles=['-o','-s','-D']
next(ax._get_lines.prop_cycler)

for file in args.files:
    data = np.loadtxt(file)

    nodes = data[:,0]
    times = data[:,1]

    nref = nodes[args.ref]
    tref = times[args.ref]

    tref = tref

    speedup = tref / times
    efficiency = speedup / nodes

    ax.plot(nodes, speedup, mstyles[i], label= '$' + getDomain(file) + '^3$',
            markeredgewidth=1, markeredgecolor='black', ms=4, linewidth=1.5, zorder=5)
    i+=1
        
nodes = [ x**3 for x in [1,2,3,4,5,7,9] ]
strnodes = [ '$'+str(x)+'^3$' for x in [1,2,3,4,5,7,9] ]
ax.set_xscale("log", nonposx='clip')
#ax.set_yscale()
#ax.set_yscale("log", nonposy='clip')

#ax.set_yticks(nodes)
#ax.set_yticklabels(np.array(nodes, dtype=int))

ax.set_xticks(nodes)
ax.set_xticklabels(strnodes)


ax.plot(nodes, np.ones(len(nodes)), "k--", label="ideal")

ax.set_xlabel("Nodes")
ax.set_ylabel("Efficiency  $E = \dfrac{t_1}{t_N}$")
plt.legend()#frameon=False)

myplt.set_font_sizes(ax)
myplt.make_grid(ax, True)

if blood:
    myplt.save_figure(fig, 3, 'weak_scale_blood.pdf')
else:
    myplt.save_figure(fig, 3, 'weak_scale_poiseuille.pdf')

plt.show()
