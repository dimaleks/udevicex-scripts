import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib.cm as cm


import  mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

import udevicex as udx
ranks  = (1, 1, 1)
domain = (32, 16, 4)
dt = 0.001
niters = 20
u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=8)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=1.0, dt=dt, power=0.5)
u.registerInteraction(dpd)

vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
u.registerIntegrator(vv)

center=(domain[0]*0.5, domain[1]*0.5)
radius=domain[1]*0.2
wall = udx.Walls.Cylinder("cylinder", center=center, radius=radius, axis="z", inside=False)
u.registerWall(wall, 0)
frozen = u.makeFrozenWallParticles(pvName="wall", walls=[wall], interaction=dpd, integrator=vv, density=8, nsteps=500)

u.setWall(wall, pv)

for p in (pv, frozen):
    u.setInteraction(dpd, p, pv)

u.setIntegrator(vv, pv)


vimposer = udx.Plugins.createImposeVelocity('vel', pv, 1, low=(0,0,0), high=domain, velocity=(2,0,0))
u.registerPlugins(vimposer)
field = udx.Plugins.createDumpAverage('field', [pv], 1, niters, (1, 1, 4), [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)
u.registerPlugins(udx.Plugins.createStats('stats', every=500))


u.run(niters+2)

if not u.isComputeTask():
    view = field[1].get_channel_view('velocity')
    
    vx = view[:,:,0,0]
    vy = view[:,:,0,1]
    
    x = np.arange(domain[0]) + 0.5
    y = np.arange(domain[1]) + 0.5
    X,Y = np.meshgrid(x,y)

rank = MPI.COMM_WORLD.Get_rank()

if rank == 1:
    def set_bcast(val):
        MPI.COMM_WORLD.Ibcast(np.array([float(val)]), root=1).wait()
    
    # Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)
    plt.subplots_adjust(bottom=0.2)
    
    cyl = plt.Circle(center, radius, color='black')
    ax.add_artist(cyl)
    
    ax.set_xlim([0,domain[0]])
    ax.set_ylim([0,domain[1]])
    ax.set_aspect('equal', 'datalim')
    
    q = ax.quiver(X, Y, view[:,:,0,0], view[:,:,0,1], vx**2 + vy**2, cmap='jet')
    q.set_clim(0,8)
    
    # Slider
    ax_diam = plt.axes([0.2, 0.1, 0.65, 0.03])
    sl_diam = Slider(ax_diam, 'Velocity', -2, 2, valinit=2.0)
    sl_diam.on_changed(lambda val: set_bcast(val))
    
    # Animation
    def animate(i):
        u.run(niters)
        q.set_UVC(vx, vy, vx**2 + vy**2)
        return q,
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=25)
    
    plt.show()
    
else:
    val = np.empty(1)
    req = MPI.COMM_WORLD.Ibcast(val, root=1)
    while True:
        u.run(niters)
        if req.Test():
            vimposer[0].set_target_velocity((val[0], 0, 0))
            req = MPI.COMM_WORLD.Ibcast(val, root=1)

