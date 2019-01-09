import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import trimesh

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    ax.set_aspect('equal')
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

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
    
    q = raw[:,5:9]
    
    return x,y,z, vx,vy,vz, wx,wy,wz, q

x,y,z, vx,vy,vz, wx,wy,wz, q = read_all('/home/alexeedm/extern/daint/scratch1600/focusing_square_rigid_massive/newcode/case_free_0___Re_50__kappa_0.22__a_160__gamma_25.7077__kbt_3__dt_0.001__f_0.15337__ry_0.48912__rz_0.489121__wy_0.0__wz_0.0/pos/sphere.txt')

t, sep = pickle.load(open('data/focusing/separatrix_Re_50_kappa_0.22__rotation_0.pckl','rb'))

dom_size = 49
h = 45.455
r = 5.0 / (0.5*h)

y = (2*y - dom_size) / h
z = (2*z - dom_size) / h

t = 0



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d', proj_type = 'ortho', autoscale_on=False)

sphere = trimesh.creation.icosphere(2, r)
color = np.array([ plt.cm.RdYlBu(i) for i in np.linspace(0, 1, num=sphere.faces.shape[0]) ])
color[:,-1] = 0.5




def draw(val):
    t = int(val)
    
    ax.clear()
    
    ax.set_ylim([-0.0,1.1])
    ax.set_zlim([-0.2,1.1])
    ax.view_init(0, 0)

    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    rot = trimesh.transformations.quaternion_matrix(q[t])
    sphere.apply_transform(trimesh.transformations.inverse_matrix(rot))
    sphere.apply_translation((y[t], y[t], z[t]))
    verts = sphere.vertices
    
    set_axes_equal(ax)
    
    ax.plot([ax.get_ylim()[0],1], [1,1], zdir='x', linewidth=2, color='black')
    ax.plot([1,1], [ax.get_zlim()[0],1], zdir='x', linewidth=2, color='black')



    ax.plot_trisurf(verts[:,1], verts[:,0], verts[:,2], triangles=sphere.faces, edgecolors=color, alpha=0.2, zorder=1)
    
    sphere.apply_translation((-y[t], -y[t], -z[t]))
    sphere.apply_transform(rot)
    
    ax.plot(y[:t], z[:t], zdir='x', linewidth=0.4, alpha=0.7, zorder=0, color='C0')
    ax.plot(sep[:,0], sep[:,1], zdir='x', linewidth=5, alpha=0.5, zorder=1, color='C1')
    ax.plot(sep[:,1], sep[:,0], zdir='x', linewidth=5, alpha=0.5, zorder=1, color='C1')
    
 
    ax.quiver(x[t]*0+1, y[t], z[t], vx[t]*0, vy[t]*2, vz[t]*2, color='red', zorder=2)
    ax.quiver(x[t]*0+1, y[t], z[t], wx[t]*0, wy[t]*1.5, wz[t]*1.5, color='green', zorder=2)
    
    
folder = 'data/focmov/'

step = 15#35000 / (30*60)

c = 0
for i in range(0, 35000, int(step)):
    
    fname = "img_%06d.png" % (c,)
    c+=1
    
    print(i)
    draw(i)

    fig.savefig(folder + fname, bbox_inches='tight', dpi=300)

#ax_sl = plt.axes([0.2, 0.025, 0.65, 0.03])
#sl = mpl.widgets.Slider(ax_sl, 'Time-step', 0, x.size-1, valinit=t, valfmt="%d")
#sl.on_changed(draw)
#
#plt.show()