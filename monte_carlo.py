# Coded for Python 2.7

import numpy as np, os, sys
from functions import *

np.random.seed(seed=1)

filename = 'results.txt'
if os.path.isfile(filename):
    os.remove(filename)
f = open(filename, 'a')
sys.stdout = f

######################

N = [100, 1000, 5000] # population sizes
B = 5000 # number of reps

deltas = [0,0.4,0.5] # used in definition of p0 below
p1 = 0.5 # success probability for stratum 1
px = 0.5 # probability of being in stratum 0

estimand = np.zeros((len(N),B)) # finite population ATE
estimator = np.zeros((len(N),B)) # estimates of ATE
standard_error = np.zeros((len(N),B)) # standard error
pscore = np.zeros((len(N),B))

for block_rand in [True,False]:
    if block_rand:
        print "\nRESULTS FOR STRATIFIED BLOCK RANDOMIZATION\n"
    else:
        print "RESULTS FOR CONDITIONALLY INDEPENDENT RANDOMIZATION\n"

    for delta in deltas:
        print "delta = %s" % delta

        for k in range(len(N)):  
            n = N[k]
            p0 = min(n**(-delta), 0.5) # success probability for stratum 0

            for b in range(B):
                X = np.random.binomial(1, px, n)
                theta = np.random.multivariate_normal(mean=[0,0.5,2,1],cov=np.eye(4),size=n)

                if block_rand:
                    obs = np.arange(n,dtype=np.int16)
                    stratum1 = obs[X==1]
                    stratum0 = obs[X==0]
                    treated = np.hstack([np.random.choice(stratum1, int(np.ceil(p1*stratum1.shape[0])), replace=False), np.random.choice(stratum0, int(np.ceil(p0*stratum0.shape[0])), replace=False)])
                    D = np.array([i in treated for i in range(n)])
                else:
                    D = np.random.rand(n) < ((X==1)*p1 + (X==0)*p0)

                W = np.vstack([np.ones(n), X, D, X*D]).T
                Y = (W * theta).sum(axis=1)

                estimand[k,b] = tau(Y,X,theta)
                estimator[k,b] = hat_tau(Y,D,X)
                standard_error[k,b] = SE(Y,D,X)
                pscore[k,b] = (D*(X==0)).sum() / float((X==0).sum())

        coverage = ((estimator-1.96*standard_error < estimand) * (estimand < estimator+1.96*standard_error)).mean(axis=1)

        print N
        print coverage
        print pscore.mean(axis=1)
        print "\n"

f.close()
