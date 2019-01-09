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
        
    return m,np.sqrt(v)

parser = argparse.ArgumentParser(description='Postprocess')
parser.add_argument('--filename', help='File with rbc trajectories', type=str, default='pos/rbc.txt')
parser.add_argument('--show', help='only show trajectories', action='store_true')
parser.add_argument('--resource-folder', help='Path to all the required files', type=str, default='./')
args = parser.parse_args()


sdf_file = 'sdf.dat'
with open(sdf_file, 'r', encoding='latin-1') as sdf:
    header = sdf.readline()
    domain = tuple( [float(v) for v in header.split(' ')] )


df = pd.read_csv(args.filename, sep='\s+', header=None)

# Cut away the last step, as it may be corrupted
data = df.values
if np.isnan(data[-1, :]).any():
    data = data[:-1, :]

laststep = np.amax(data[:,1])
all_but_last_idx = data[:,1] < (laststep - 1e-6)
data = data[ all_but_last_idx, : ]

nsteps = np.unique(data[:,1]).size
per_cell = data[data[:,0].argsort(kind='stable')]
ncells = per_cell.shape[0] / nsteps

print('Total # of points: %d, time-steps: %d, ncells %f' % (per_cell.shape[0], nsteps, ncells))


if nsteps < 2000:
    np.savetxt('loglike.txt', [-1e10] )
    quit()

if np.abs(ncells - int(ncells)) > 1e-8:
    print('POSSIBLY FAILED')

inclinations = []
for c in range(round(ncells)):
    
    traj = remove_nans(per_cell[c*nsteps : (c+1)*nsteps])
    
    x = correct_period(traj[:, 2], domain[0])
    y = correct_period(traj[:, 3], domain[1])

    startid = int(y.size/3)
    
    try:
        if x[-1] - x[startid] > 2*domain[0]:
            inclinations += [ (y[-1] - y[startid]) / (x[-1] - x[startid]) ]
    except:
        pass
       
    if args.show:
        xraw = traj[::2, 2]
        yraw = traj[::2, 3]

        dy = np.abs(yraw[1:] - yraw[:-1])
        yraw[1:][ dy > 10 ] = float('nan')
        
        dx = np.abs(xraw[1:] - xraw[:-1])
        xraw[1:][ dx > 10 ] = float('nan')
        
        import matplotlib.pyplot as plt
        plt.plot(xraw, yraw, linewidth=0.2) #, linewidth = (1.0 if fallen else 0.2))

m, std = mean_std(inclinations)
print('Average %f, std %f' % (-m, std))

if args.show:
    plt.show()
else:
    np.savetxt('loglike.txt', [-m] )

