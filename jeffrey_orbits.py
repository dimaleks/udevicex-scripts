#!/usr/bin/env python

import sys
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


def set_pgf_backend():
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-linux/'

    pgf_with_custom_preamble = {
        'pgf.texsystem' : 'pdflatex',
        "text.usetex": True,    # use inline math for ticks
        "font.size": 10,
        "pgf.preamble": [r'\usepackage{txfonts}',
                         r'\usepackage{amssymb}',
                         r'\DeclareUnicodeCharacter{2212}{-}']
        }
    mpl.rcParams.update(pgf_with_custom_preamble)
    plt.switch_backend('pgf')
    
    
def trajectory(fname):
    
    lines = np.loadtxt(fname)
    
    c = lines[:, 5]
    s = lines[:, 7]
    
    phi = 2*np.arctan2(s, c)
    t = lines[:, 1]
    
    phi = phi[::]
    t = t[::]
    
    phi = np.arctan(np.tan(phi))
    
    fromid, toid = 3317, 3317 + int(3*2848/2) + 4
    
    return phi[fromid : toid], t[fromid : toid] - t[fromid]


def fit_traj(t):
    a = 3.0
    b = 5.0
    G = 0.1
    
    ref = np.arctan( b/a * np.tan(a*b*G * (t-0.5) / (a**2 + b**2)) )
    
    return ref
    

def dump_plots(phi, t, ref, tref):
    
    G = 0.1
    fig = plt.figure()
    
    plt.plot(tref / G, ref, label="Analytical",  color="C7", linewidth=1.5)
    plt.plot(t[::55] / G, -phi[::55], 'D', label="Simulation", color="C2", markeredgewidth=0.5, markeredgecolor='black', ms=2, zorder=2)
    
    plt.tick_params(axis='both', which='major', labelsize=8)

    plt.xlabel(r'$t / \dot{\gamma}$', fontsize=9)
    plt.ylabel(r'$\phi$', fontsize=9)
    
    fig.tight_layout()
    
    ax=plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.grid()
    #plt.legend(fontsize=14)

    fig.tight_layout()
    plt.show()
    
    targetw = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(targetw, h * (targetw/w))
    fig.savefig("/home/alexeedm/udevicex-scripts/media/jeffrey.pdf", bbox_inches='tight')


def main():
    
    #fname = "/home/alexeedm/udevicex/apps/udevicex/jeffrey_pos/all.txt"
    fname = "/home/alexeedm/extern/daint/scratch/jeffrey/case_4/pos/ellipsoid.txt"
    
    phi, t = trajectory(fname)
    
    tref = np.linspace(t[0], t[-1], t.size*5)
    ref = fit_traj(tref)
        
    dr = ref[1:] - ref[:-1]
    ref[1:][ dr < -1 ] = float('nan')
    
    pickle.dump((phi, t, tref, ref), open('/home/alexeedm/papers/udevicex-cpc-paper/data/jeffery/data.pckl', 'wb'))


#    plt.switch_backend('Qt5Agg')
    set_pgf_backend()

    dump_plots(phi, t, ref, tref)


if __name__ == "__main__":
    main()
