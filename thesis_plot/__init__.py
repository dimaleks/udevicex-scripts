#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:26:02 2018

@author: alexeedm
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

preamble = [ r'\usepackage{amssymb,amsmath}',
             r'\usepackage{siunitx}',
             r'\usepackage{palatino}',
             r'\usepackage{eulervm}',
             r'\usepackage{bm}',
             r'\renewcommand{\sffamily}{\rmfamily}' ]

def set_pgf_backend():
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-linux/'

    custom_rc = {
        'pgf.texsystem' : 'pdflatex',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': 10,
        'legend.fontsize': 8,
        'axes.titlesize' : 9
        }
    
    custom_rc['pgf.preamble'] = preamble
    mpl.rcParams.update(custom_rc)
    plt.switch_backend('pgf')
    
def make_grid(ax, minor=False):
    ax.grid()
    ax.grid(which='major', linewidth=0.5)
    if minor:
        ax.grid(which='minor', linewidth=0.05, alpha=0.5)

def set_qt_backend():
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-linux/'

    custom_rc = {
        'pgf.texsystem' : 'pdflatex',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': 10,
        'legend.fontsize': 8,
        'axes.titlesize' : 9
        }
    custom_rc['text.latex.preamble'] = preamble
    mpl.rcParams.update(custom_rc)
    plt.switch_backend('Qt5Agg')
    
def set_font_sizes(ax):
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.get_label().set_fontsize(9)
    ax.yaxis.get_label().set_fontsize(9)

def set_figure_size(fig, width):
    
    #width *= 1.25
    
    fig.tight_layout()
    w, h = fig.get_size_inches()
    fig.set_size_inches(width, h * (width/w))
    fig.tight_layout()

def save_figure(fig, width, fname, **kwarg):
    set_figure_size(fig, width)
    fig.savefig('/home/alexeedm/papers/thesis/figures/' + fname, bbox_inches='tight', width=width, **kwarg)
    