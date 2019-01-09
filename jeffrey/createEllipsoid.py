#!/usr/bin/env python
import sys

def createEllipsoid(density, axes, niters):
    import udevicex as udx
    
    def recenter(coords, com):
        coords = [[r[0]-com[0], r[1]-com[1], r[2]-com[2]] for r in coords]
        return coords

    dt = 0.001
    axes = tuple(axes)

    ranks  = (1, 1, 1)
    fact = 3
    domain = (fact*axes[0], fact*axes[1], fact*axes[2])
    
    u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log', no_splash=True)
    
    dpd = udx.Interactions.DPD('dpd', 1.0, a=10.0, gamma=10.0, kbt=0.5, dt=dt, power=0.5)
    vv = udx.Integrators.VelocityVerlet('vv', dt=dt)
    
    coords = [[-axes[0], -axes[1], -axes[2]],
              [ axes[0],  axes[1],  axes[2]]]
    com_q = [[0.5 * domain[0], 0.5 * domain[1], 0.5 * domain[2],   1., 0, 0, 0]]
    
    fakeOV = udx.ParticleVectors.RigidEllipsoidVector('OV', mass=1, object_size=len(coords), semi_axes=axes)
    fakeIc = udx.InitialConditions.Rigid(com_q=com_q, coords=coords)
    belongingChecker = udx.BelongingCheckers.Ellipsoid("ellipsoidChecker")
    
    pvEllipsoid = u.makeFrozenRigidParticles(belongingChecker, fakeOV, fakeIc, dpd, vv, density, niters)
    
    if pvEllipsoid:
        frozenCoords = pvEllipsoid.getCoordinates()
        frozenCoords = recenter(frozenCoords, com_q[0])
    else:
        frozenCoords = [[]]

    if u.isMasterTask():
        return frozenCoords
    else:
        return None

if __name__ == '__main__':

    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', type=float, default=8)
    parser.add_argument('--axes', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('out', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    coords = createEllipsoid(args.density, args.axes, args.niters)

    if coords is not None:
        
        if '.xyz' in args.out.name:
            print(len(coords), file=args.out)
            print('#Created by uDeviceX: ellipsoid with axes %s and density %f' % (str(args.axes), args.density),  file=args.out)
        np.savetxt(args.out, coords)
    
# nTEST: rigids.createEllipsoid
# set -eu
# cd rigids
# rm -rf pos.txt pos.out.txt
# pfile=pos.txt
# udx.run --runargs "-n 2"  ./createEllipsoid.py --axes 2.0 3.0 4.0 --density 8 --niter 1 --out $pfile > /dev/null
# cat $pfile | sort > pos.out.txt

