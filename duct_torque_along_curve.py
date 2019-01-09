import numpy as np
import matplotlib.pyplot as plt
import pickle
import thesis_plot as myplt

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kr

def val_along_curve_nongp(vy, sy, vz, sz, coo, normal=True):
    
    res = np.empty(vy.shape)
    res[0] = 0.0
    res[-1] = 0.0
    
    err = np.empty(vy.shape)
    err[0] = 0.0
    err[-1] = 0.0
    
    
    for i in range(1, res.shape[0] - 1):
        direction = 0.5 * ( (coo[i] - coo[i-1]) + (coo[i+1] - coo[i]) )
        direction /= np.linalg.norm(direction)        
        vector = np.array([vy[i], vz[i]])


        if normal:
            res[i] = np.dot(vector, direction )
        else:
            res[i] = np.linalg.norm( vector - np.dot( vector, direction ) * direction )

        err[i] = np.sqrt( sy[i]**2 + sz[i]**2 )
        
    return res, err

def val_along_curve(gpY, gpZ, coo, normal=True):
    
    vy, sy = gpY.predict( np.atleast_2d(coo), return_std=True )
    vz, sz = gpZ.predict( np.atleast_2d(coo), return_std=True )
    
    return val_along_curve_nongp(vy, sy, vz, sz, coo, normal)
        
def draw_value_along_curve(t, f, sigma, ls):
    plt.plot(t / np.max(t), f, ls, color='black')
    plt.fill_between(t / np.max(t), f - sigma, f + sigma, color='red', alpha=0.5, linewidth=0)
    #plt.title(title)
    
    #plt.axes().set_ylim(limits)
    
    plt.ylabel(r'$C_l^{||}$')
    plt.xlabel(r'$t$')
    
    myplt.set_font_sizes(plt.gca())
    myplt.make_grid(plt.gca())
    
def gaussian_fit(x, y, err):
        
    err[np.isnan(err)] = 1e-1
    kernel = kr.Matern(length_scale=0.2, nu=2.0) 
    #kernel = kr.Matern(length_scale=0.15, nu=2.0)

    gp = GaussianProcessRegressor(kernel=kernel, alpha = 3*err**2, n_restarts_optimizer=2)
    
    try:
        gp.fit(x, y)
    except:
        print(y)          
    
    return gp


kappa = 0.22
Re = 100
folder = 'data/focusing/'
fname = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '__rotation_0.pckl'

myplt.set_pgf_backend()
fig = plt.figure()

# free rotation
fy, fz = pickle.load(open(folder + 'force_' + fname,'rb'))
t, sep = pickle.load(open(folder + 'separatrix_' + fname,'rb'))
f, fsigma = val_along_curve(fy, fz, sep)
#print(f, fsigma)

draw_value_along_curve(t, f, fsigma, '-')

# modified
moddata = pickle.load(open(folder + 'modrotation'+str(int(Re))+'.pckl', 'rb'))
#print(moddata)

coo = moddata[:,0:2]

idx = np.argsort(np.arctan2(coo[:,1], coo[:,0]))
coo = coo[idx, :]
fym, eym, fzm, ezm = [ moddata[idx, i] for i in [4,5,6,7] ] 

gpym = gaussian_fit(coo, fym, eym)
gpzm = gaussian_fit(coo, fzm, ezm)

fmod, mods = val_along_curve(gpym, gpzm, coo)

lengths = np.empty(coo.shape[0])
lengths[0] = 0.0
for i in range(1, coo.shape[0]):
    lengths[i] = lengths[i-1] + np.sqrt( np.dot(coo[i-1] - coo[i], coo[i-1] - coo[i]) )
    
print(lengths)

draw_value_along_curve(lengths, fmod, mods, ':')

myplt.save_figure(fig, 3.5, 'square/mod_rotation_Re_100.pdf')






