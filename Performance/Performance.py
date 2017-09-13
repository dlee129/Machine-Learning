import numpy as np
import time

def compute_euclidean_loop(X):
    distance = np.zeros((len(X),len(X)), dtype = float)
    for r in range(len(X)):
        for c in range(len(X)):
            distance[r][c] = np.linalg.norm(X[r]-X[c])

    return distance

def compute_euclidean_cool(X):
    distance_cool = np.zeros((len(X), len(X)), dtype=float)
    distance_cool[:] = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    return distance_cool

print 'starting running .....'
np.random.seed(100)
params = range(10,51,10)   # different param setting
nparams = len(params)       # number of different parameters

perf_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
perf_cool = np.zeros([10,nparams])

counter = 0

for ncols in params:
    nrows = ncols * 10

    print "matrix dimensions: ", nrows, ncols

    for i in range(10):
        X = np.random.randint(0,20,[nrows,ncols])   # random matrix
                                                    # you need to use random.rand(...) for float matrix

        st = time.time()
        euclidean_loop = compute_euclidean_loop(X)
        et = time.time()
        perf_loop[i,counter] = et - st              # time difference

        st = time.time()
        euclidean_cool = compute_euclidean_cool(X)
        et = time.time()
        perf_cool[i,counter] = et - st

    counter = counter + 1

mean_loop = np.mean(perf_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
mean_cool = np.mean(perf_cool, axis = 0)

std_loop = np.std(perf_loop, axis = 0)      # standard deviation
std_cool = np.std(perf_cool, axis = 0)

import matplotlib.pyplot as plt
plt.errorbar(params, mean_loop[0:nparams], yerr=std_loop[0:nparams], color='red',label = 'Loop Solution')
plt.errorbar(params, mean_cool[0:nparams], yerr=std_cool[0:nparams], color='blue', label = 'Matrix Solution')
plt.xlabel('Number of Cols of the Matrix')
plt.ylabel('Running Time (Seconds)')
plt.legend()
plt.savefig('EuclideanDistance.pdf')
# plt.show()    # uncomment this if you want to see it right way

print "result is written to EuclideanDistance.pdf"
