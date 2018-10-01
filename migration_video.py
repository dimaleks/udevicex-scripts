import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def read_all(fname):
    raw = np.loadtxt(fname)
    
    x = raw[:, 2]
    y = raw[:, 3]
    z = raw[:, 4]
    
    vx = raw[:, 9]
    vy = raw[:, 10]
    vz = raw[:, 11]
    
    wx = raw[:, 12]
    wy = raw[:, 13]
    wz = raw[:, 14]
    
    return x,y,z, vx,vy,vz, wx,wy,wz

x,y,z, vx,vy,vz, wx,wy,wz = read_all('/home/alexeedm/extern/daint/scratch1600/focusing_square_rigid_massive/newcode/case_free_0___Re_50__kappa_0.22__a_160__gamma_25.7077__kbt_3__dt_0.001__f_0.15337__ry_0.48912__rz_0.489121__wy_0.0__wz_0.0/pos/sphere.txt')

dom_size = 49
h = 45.455
r = 5.0 / (0.5*h)

y = (2*y - dom_size) / h
z = (2*z - dom_size) / h

t = 0


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False)

def draw(val):
    t = int(val)
    
    ax.clear()
    
    ax.set_xlim([-0.3,0.8])
    ax.set_ylim([-0.3,0.8])
    ax.set_aspect('equal', 'datalim')
    
    ax.plot(y[:t], z[:t], linewidth=0.2, alpha=0.5, zorder=0)
    
    ax.add_artist(plt.Circle((y[t], z[t]), r, color='black', alpha=0.1, zorder=1))
    ax.quiver(y[t], z[t], vy[t], vz[t], color='red', zorder=2)
    ax.quiver(y[t], z[t], wy[t], wz[t], color='green', zorder=2)
        
    
ax_sl = plt.axes([0.2, 0.025, 0.65, 0.03])
sl = mpl.widgets.Slider(ax_sl, 'Time-step', 0, x.size-1, valinit=t, valfmt="%d")
sl.on_changed(draw)

plt.show()