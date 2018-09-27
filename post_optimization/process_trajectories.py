import numpy as np
import pandas as pd
import argparse

def remove_nans(x):
    return x[ ~np.isnan( np.sum(x, axis=1) ) ]

def correct_period(x, period):
    dx = x[1:] - x[:-1]
    
    xadd = np.zeros(x.size)
    xadd[1:][ dx >  period*0.5] -= period
    xadd[1:][ dx < -period*0.5] += period
    return x + np.cumsum(xadd)

def mean_std(vals):
    
    npvals = np.array(vals)
    m = np.mean(npvals)
    v = np.var(npvals) / npvals.size
        
    return m,v

parser = argparse.ArgumentParser(description='Postprocess')
parser.add_argument('--filename', help='File with rbc trajectories', type=str, default='pos/rbc.txt')
parser.add_argument('--domain', help='Domain size', type=float, nargs=3, default=[32, 56, 60])

args = parser.parse_args()


df = pd.read_csv(args.filename, sep='\s+', header=None)

nsteps = np.unique(df.values[:,1]).size
per_cell = df.values[df.values[:,0].argsort(kind='stable')]

print('Total # of points: %d, time-steps: %d, ncells %f' % (per_cell.shape[0], nsteps, per_cell.shape[0] / nsteps))

inclinations = []
for c in range(round(per_cell.shape[0] / nsteps)):
    
    traj = remove_nans(per_cell[c*nsteps : (c+1)*nsteps])
    
    x = correct_period(traj[:, 2], args.domain[0])
    y = correct_period(traj[:, 3], args.domain[1])
    
#    import matplotlib.pyplot as plt
#    plt.plot(x, y)
#    plt.show()
    
    inclinations += [ (y[-1] - y[0]) / (x[-1] - x[0]) ]
    

m, std = mean_std(inclinations)
print('Mean y/x = %f, standard deviation %f' % (m, std))

np.savetxt('loglike.txt', [m] )
    
    
