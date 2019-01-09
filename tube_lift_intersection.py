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

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kr


def coefficient(frc, rho, u, r, R):
    return frc / (rho * u**2 * (2*r)**4 / (2*R)**2)

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
    
    Uavg = (2.0 * R)**2 * rho*f / (32*mu) # average!!!
    
    positions = np.linspace(0.0, 0.7, 8)
    
    Cls = [0]
    err_Cls = [0]
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
        
    return np.array(x), np.array(Cls), np.array(err_Cls)

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
        
        m = re.search(r'case_(.*?)_(.*?)_(.*?)_.*?__(.*?)_(.*?)_.*?__', c.split('/')[-1])
        f, lbd, Y, a, gamma = [ float(v) for v in m.groups() ]
        
        s = pickle.load( open('data/visc_' + str(a) + '_0.5_backup.pckl', 'rb') )
        mu = s(gamma)
                
        x, f, err = get_forces(c, kappa, f, mu)
        alldata.append( (x, f, err, Y, lbd) )

    return alldata

def process_data(data):

    res = []
    
    for x, f, err, Y, lbd in data:
        gp = gaussian_fit(np.atleast_2d(x).T, f, err)
        
        res.append( (x, f, err, gp, Y, lbd) )
        
    return res




def mean_plusminus_std(gp, x):
    
    y, std = gp.predict(np.atleast_2d(np.array([x])), return_std=True)
    return y[0]-std[0], y[0]+std[0]

def intersection(gp):
    
    fval = lambda x : gp.predict(np.atleast_2d(np.array([x])), return_std=False)[0]
    x0 = fsolve(fval, 0.65)
    
    ferrlo = lambda x : mean_plusminus_std(gp, x)[0]
    xlo = fsolve(ferrlo, x0)

    ferrhi = lambda x : mean_plusminus_std(gp, x)[1]
    xhi = fsolve(ferrhi, x0)

    return x0[0], x0[0]-xlo[0], xhi[0]-x0[0]

def dump_plots(alldata, Re, kappa):
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(60, 40))
    #plt.suptitle(r'$Re = ' + str(Re) + r'$, $\kappa = ' + str(kappa) + r'$')
    
    axes = axes.flatten()
    
    i=0
    for x, data, err, gp, Y, lbd in alldata:            
        label = r'$Ca =' + str(Y) + '$, $\lambda = ' + str(lbd) + r'$'
        
        gpx = np.linspace(0.0, np.max(x), 100)
        axes[i].errorbar(x, data, yerr=3.0*err, fmt='D', 
            label=label, color='C2', markeredgewidth=0.75, markeredgecolor='black', ms=3, zorder=5)
        
#        print x
#        print data
#        print err
#        print ""
        
        y_fit, sigma = gp.predict(np.atleast_2d(gpx).T, return_std=True)
        axes[i].plot(gpx, y_fit, color='black', zorder=4)
        axes[i].fill_between(gpx, y_fit - 2*sigma, y_fit + 2*sigma, color='red', alpha=0.5, linewidth=0, zorder=3)
        
        axes[i].grid()
        axes[i].legend(loc=3, fontsize=7)
        
        if i % 4 == 0:
            axes[i].set_ylabel(r'$C_{l}$')
            
        if i >= 8:
            axes[i].set_xlabel(r'$y/R$')
            
        myplt.set_font_sizes(axes[i])
        
        i=i+1
        
    
#    plt.xlabel('y/R', fontsize=16)
#    plt.ylabel('Cl', fontsize=16)


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01, wspace=0.01)

    #plt.show()
    myplt.save_figure(fig, 7, 'soft_circle/tube_lift_soft__Re_' + str(Re) + '_kappa_' + str(kappa) + '.pdf')
    plt.close(fig)

#%%

myplt.set_pgf_backend()
folder = "/home/alexeedm/extern/daint/project/alexeedm/focusing_soft/"
Re = 200
kappa = 0.15

plt.ioff()

for Re in [50, 100, 200]:
    for kappa in [0.15, 0.22, 0.3]:
        
        Re = 200
        kappa = 0.22
        
        print(Re, kappa)
        print("")
        print("")
        
        np.set_printoptions(linewidth = 200)
        #data = get_data(folder + 'newcase_' + str(Re) + '_' + str(kappa) + '/', Re, kappa)
        #pickle.dump(data, open('data/soft_' + str(Re) + '_' + str(kappa) + '.pckl', 'wb'))
        data = pickle.load(open('data/soft_' + str(Re) + '_' + str(kappa) + '.pckl', 'rb'))
        
        #%%
        def gaussian_fit(x, y, err):
            
            err[np.isnan(err)] = 1e-1
            #kernel = kr.RBF(length_scale=0.2)
            kernel = kr.Matern(100, (0.5, 1e3), nu=2.5) 
#            kernel = kr.ConstantKernel(1.0, (1e-3, 1e3)) * kr.RBF(1, (1e-2, 1e2)) #
            #kernel = kr.Matern(length_scale=0.15, nu=2.0)
            
            nerr = err*1.0
            nerr[-1] *= 0.25
            gp = GaussianProcessRegressor(kernel=kernel, alpha = nerr**2, n_restarts_optimizer=20, normalize_y=True)
            
            #print y
            gp.fit(x, y)
            print(gp.kernel_)
        
            
            return gp
            
        processed = process_data(data)
        dump_plots(processed, Re, kappa)
        
        
        
        fig = plt.figure()
        plt.title(r'$Re = ' + str(Re) + r'$, $\kappa = ' + str(kappa) + r'$')
        
        intersections = np.empty((0, 4))
        for x, d, err, gp, Y, lbd in processed:
            x0, elo, ehi = intersection(gp)
            
            intersections = np.vstack( (intersections, np.array([x0, 0.5*(elo+ehi), Y, lbd])) )    
        
        #%%
        for lbd, style in [(1.0, '--D'), (5.0, '--o'), (25.0, '--s')]:
            idxs = np.where(intersections[:,3] == lbd)
            plt.errorbar(intersections[idxs, 2], intersections[idxs, 0], yerr=intersections[idxs, 1], label=r'$\lambda=' + str(lbd) + r'$',
              fmt=style, markeredgewidth=0.75, markeredgecolor='black', ms=3, zorder=2, linewidth=1.0)
            
        #TODO: eqiulibrium vs size and vs Reynolds 
        
        plt.legend()
        
        plt.xscale('log')
        plt.xlabel(r'$Ca$')
        plt.ylabel(r'$C_0/R$')
        
        myplt.make_grid(plt.gca())
        
        
        plt.tight_layout()
        myplt.save_figure(fig, 3, 'soft_circle/tube_lift_soft__intersection_' + str(Re) + '_kappa_' + str(kappa) + '.pdf')
        plt.close(fig)

        a= alsdfkja

