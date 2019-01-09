#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:39:29 2018

@author: alexeedm
"""
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pickle
from scipy.optimize import fsolve
import thesis_plot as myplt
import pandas as pd


from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kr


def coefficient(frc, rho, u, r, R):
    return frc / (rho * u**2 * (2*r)**4 / (2*R)**2)

def process_rotation(w, U, H):
    return w / (U/H)

def mean_err_cut(vals):
    npvals = np.array(vals[20:]).astype(np.float)
    
    npvals = npvals[ np.where(npvals < 10000) ]
    
    m = np.mean(npvals)
    v = np.var(npvals) / npvals.size
    
    return m,v

def get_forces(case, kappa, f, mu):
    prefix = ""    
    rho = 8.0
    r = 5
    R = r/kappa
    
    Uavg = (2.0 * R)**2 * rho*f / (32*mu)
    
    if kappa > 0.17:
        positions = np.linspace(0.0, 0.7, 8)
    else:
        positions = np.linspace(0.0, 0.8, 9)

    
    print('Re = ' + str(Uavg*rho*2*R / mu))
    
    Cls = [0]
    ws = [0]
    err_Cls = [0]
    err_ws = [0]
    x = [0]
    
    for pos in positions:
        if pos < 0.0001:
            continue
        
        strpos = "%.1f" % pos
        full_folder = prefix + case + strpos
        
        files = sorted(glob.glob(full_folder + "/pinning_force/*.txt"))
        lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
        
        if len(lines) > 0:
        
            fy = [ l.split()[3] for l in lines ]
        
            (my, vy) = mean_err_cut(fy)
            Cls.append(coefficient(my, rho, Uavg, r, R))
            err_Cls.append(coefficient(math.sqrt(vy), rho, Uavg, r, R))
            x.append(pos)
            
            
        # also rotations
        files = sorted(glob.glob(full_folder + "/pos/*.txt"))
        df = pd.read_csv(files[0], sep='\s+', header=None) 
        w = df.values[:,14]
        w = w[ np.where(~np.isnan(w)) ]
        
        (mw, vw) = mean_err_cut(w)
        ws.append(process_rotation(mw, Uavg, 2*R))
        err_ws.append(process_rotation(math.sqrt(vw), Uavg, 2*R))
        
        
    return np.array(x), np.array(Cls), np.array(err_Cls), np.array(ws), np.array(err_ws)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def get_data(folder, Re, kappa):
    print( list(map(float, list(filter(lambda x: is_number(x), glob.glob(folder + '*_0.1/')[0].split('_'))))) )
    cases01 = sorted(  glob.glob(folder + "*_0.1/"), key = lambda v : list( map(float, list(filter(lambda x : is_number(x), v.split('_')))) ) )
    
    #print cases01
        
    cases = [ re.match(r'(.*)0.1/', c01).group(1) for c01 in cases01 ]    
    
#    fs = get_forces("/home/alexeedm/extern/daint/scratch/focusing_liftparams/case_newcode_ratio_5_0.05177__110_25_2.0__", kappa, 0.05177, 24.77)
#    alldata = [ fs + (r'$\hat Y = -\infty$, $\lambda = \infty$', '-D') ]
    alldata = [ ]
    
    for c in cases[0:]:
        print(c)
        
        m = re.search(r'case_(.*?)_(.*?)_.*?__(.*?)_(.*?)_.*?__', c.split('/')[-1])
        f, rot, a, gamma = [ float(v) for v in m.groups() ]
        
        s = pickle.load( open('data/visc_' + str(a) + '_0.5_backup.pckl', 'rb') )
        mu = s(gamma)
                
        x, f, err, w, errw = get_forces(c, kappa, f, mu)
        
        if kappa > 0.25:
            x = x[:-1]
            f = f[:-1]
            err = err[:-1]
        
        alldata.append( (x, f, err, Re, kappa, rot, w, errw) )

    return alldata

def process_data(data):

    res = []
    
    for x, f, err, Re, kappa, rot, w, errw in data:
        gp = gaussian_fit(np.atleast_2d(x).T, f, err)
        
        res.append( (x, f, err, gp, Re, kappa, rot, w, errw) )
        
    return res




def mean_plusminus_std(gp, x):
    
    y, std = gp.predict(np.atleast_2d(np.array([x])), return_std=True)
    return y[0]-std[0], y[0]+std[0]

def intersection(gp):
    
    fval = lambda x : gp.predict(np.atleast_2d(np.array([x])), return_std=False)[0]
    x0 = fsolve(fval, 0.75)
    
    ferrlo = lambda x : mean_plusminus_std(gp, x)[0]
    xlo = fsolve(ferrlo, x0)

    ferrhi = lambda x : mean_plusminus_std(gp, x)[1]
    xhi = fsolve(ferrhi, x0)

    return x0[0], x0[0]-xlo[0], xhi[0]-x0[0]

def dump_plots(alldata):
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey='row', figsize=(50, 38))
    #plt.suptitle(r'$Re = ' + str(Re) + r'$, $\kappa = ' + str(kappa) + r'$')
    
    axes = axes.flatten()
    
    i=0
    for x, data, err, gp, Re, kappa, rot, w, errw in alldata:
        label = (r'$Re = '+ str(Re) + r', \kappa = ' + str(kappa) + r'$')  if rot == 0.0 else ''
        fmt = 'D' if rot == 0.0 else 'o'
        
        gpx = np.linspace(0.0, np.max(x), 100)
        axes[i].errorbar(x, data, yerr=3.0*err, fmt=fmt, 
            label=label, color='C2', markeredgewidth=0.75, markeredgecolor='black', ms=3, zorder=3)
        
#        print x
#        print data
#        print err
#        print ""
        
        y_fit, sigma = gp.predict(np.atleast_2d(gpx).T, return_std=True)
        axes[i].plot(gpx, y_fit, linewidth=0.75, color='black')
        axes[i].fill_between(gpx, y_fit - sigma, y_fit + sigma, color='red', alpha=0.5, linewidth=0, zorder=2)
        
        axes[i].grid()
        leg = axes[i].legend(loc=3, fontsize=7, handlelength=0,  handletextpad=0, markerscale=0)
        for item in leg.legendHandles:
            item.set_visible(False)
        
        if i % 3 == 0:
            axes[i].set_ylabel(r'$C_{l}$')
            
        if i >= 6:
            axes[i].set_xlabel(r'$y/R$')
            
        myplt.set_font_sizes(axes[i])
        myplt.make_grid(axes[i])
        
        if rot == 1.0:
            i=i+1
        
    
#    plt.xlabel('y/R', fontsize=16)
#    plt.ylabel('Cl', fontsize=16)


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01, wspace=0.01)

    #plt.show()
    myplt.save_figure(fig, 6.5, 'tube_lift_rigid.pdf')
    plt.close(fig)

#%%

myplt.set_pgf_backend()
folder = "/home/alexeedm/extern/daint/scratch/focusing_rigid/"
Re = 200
kappa = 0.15

plt.ioff()

data = []
for Re in [50, 100, 200]:
    for kappa in [0.15, 0.22, 0.3]:
        
        print(Re, kappa)
        print("")
        print("")
        
        np.set_printoptions(linewidth = 200)
        data += get_data(folder + 'goodcase_' + str(Re) + '_' + str(kappa) + '/', Re, kappa)
#%%
def gaussian_fit(x, y, err):
    
    err[np.isnan(err)] = 0.0
    noise = np.mean(err)
    
    #kernel = kr.RBF(length_scale=0.2)
    kernel = kr.Matern(length_scale=0.2, nu=15.0)
    
    gp = GaussianProcessRegressor(kernel=kernel, alpha = (3*err)**2.0, n_restarts_optimizer=20)
    
    #print y
    try:
        gp.fit(x, y)
    except:
        print(y)          
    
    return gp

processed = process_data(data)

#%%
dump_plots(processed)

intersections = np.empty((0, 5))
for x, d, err, gp, Re, kappa, rot, w, errw in processed:
    x0, elo, ehi = intersection(gp)
    
    intersections = np.vstack( (intersections, np.array([x0, 0.5*(elo+ehi), Re, kappa, rot])) )    
    
print(intersections)
    
pickle.dump((processed, intersections), open('data/tube_rigid_lifts.pckl', 'wb'))

#%%
def plot_equilibrium(rot, suffix):
    # x is kappa, free rotation
    fig = plt.figure()
    for Re, style in [(50, '--D'), (100, '--o'), (200, '--s')]:
        idxs = np.where( (intersections[:,4] == rot) & (intersections[:,2] == Re) )
        plt.errorbar(intersections[idxs, 3], intersections[idxs, 0], yerr=intersections[idxs, 1], label=r'$Re = ' + str(Re) + r'$',
                     fmt=style, markeredgewidth=0.75, markeredgecolor='black', ms=3, zorder=2, linewidth=1.0)
    
    plt.legend(labelspacing=0.2)
    
    #plt.xscale('log')
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$Eq/y$')
    myplt.set_font_sizes(plt.gca())
    myplt.make_grid(plt.gca())
    if rot == 1:
        y = plt.ylim()
        plt.ylim(y[0], 0.46)
    
    plt.tight_layout()
    myplt.save_figure(fig, 3, 'tube_lift_rigid_' + suffix +'_kappa.pdf')
    plt.close(fig)
    
    
    # x is Re, no rotation
    fig = plt.figure()
    for kappa, style in [(0.3, '--D'), (0.22, '--o'), (0.15, '--s')]:
        idxs = np.where( (intersections[:,4] == rot) & (intersections[:,3] == kappa) )
        plt.errorbar(intersections[idxs, 2], intersections[idxs, 0], yerr=intersections[idxs, 1], label=r'$\kappa = ' + str(kappa) + r'$',
                     fmt=style, markeredgewidth=0.75, markeredgecolor='black', ms=3, zorder=2, linewidth=1.0)
    
    plt.legend(labelspacing=0.2)
    
    #plt.xscale('log')
    plt.xlabel(r'$Re$')
    plt.ylabel(r'$Eq/R$')
    myplt.set_font_sizes(plt.gca())
    myplt.make_grid(plt.gca())
    
    if rot == 0:
        y = plt.ylim()
        plt.ylim(y[0], 0.75)
    
    plt.tight_layout()
    myplt.save_figure(fig, 3, 'tube_lift_rigid_' + suffix +'_re.pdf')
    plt.close(fig)

plot_equilibrium(1, 'norot')
plot_equilibrium(0, 'free')


