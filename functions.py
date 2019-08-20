import numpy as np, math

def tau(Y,X,theta):
    """
    Finite population ATE.
    """
    n = Y.shape[0]
    W1 = np.vstack([np.ones(n), X, np.ones(n), X]).T 
    W0 = np.vstack([np.ones(n), X, np.zeros(n), np.zeros(n)]).T
    Y1 = (W1 * theta).sum(axis=1)
    Y0 = (W0 * theta).sum(axis=1)
    
    return Y1.mean() - Y0.mean()
    
def hat_tau(Y,D,X):
    """
    Estimator for ATE.
    """
    n = float(Y.shape[0])

    t = 0

    for x in [0,1]:
        pscore_sum = (D*(X==x)).sum()
        if pscore_sum == 0: pscore_sum = 1
        pscore_sum_not = ((1-D)*(X==x)).sum()
        if pscore_sum_not == 0: pscore_sum_not = 1
        t += (X==x).sum() / n * ( (Y*D*(X==x)).sum() / pscore_sum - (Y*(1-D)*(X==x)).sum() / pscore_sum_not )

    return t

def SE(Y,D,X):
    """
    Standard error for ATE.
    """
    n = float(Y.shape[0])
    pscore = [(D*(X==0)).sum() / float((X==0).sum()), (D*(X==1)).sum() / float((X==1).sum())]
    a_n = min(pscore[0]*(1-pscore[0]), pscore[1]*(1-pscore[1]))
    s2 = 0

    for x in [0,1]:
        pscore_sum = (D*(X==x)).sum()
        pscore_sum_not = ((1-D)*(X==x)).sum()

        if pscore[x] == 0:
            pscore_sum = 1
            ap = 1
            ap_not = 0
        elif pscore[x] == 1:
            pscore_sum_not = 1
            ap = 0
            ap_not = 1
        else:
            ap = a_n / pscore[x]
            ap_not = a_n / (1-pscore[x])

        mu = [(Y*(1-D)*(X==x)).sum() / pscore_sum_not, (Y*D*(X==x)).sum() / pscore_sum]
        treat = ap * ((Y-mu[1])**2*D*(X==x)).sum() / pscore_sum
        untreat = ap_not * ((Y-mu[0])**2*(1-D)*(X==x)).sum() / pscore_sum_not
        s2 += (X==x).sum() / n * ( treat + untreat )

    if a_n == 0:
        return 1000
    else:
        return math.sqrt(s2 / a_n / n)

def bias(Y,X,theta,p0,p1):
    """
    Asymptotic difference between \hat\sigma_n^2 and \sigma_n^2.
    """
    n = Y.shape[0]
    W1 = np.vstack([np.ones(n), X, np.ones(n), X]).T 
    W0 = np.vstack([np.ones(n), X, np.zeros(n), np.zeros(n)]).T
    Y1 = (W1 * theta).sum(axis=1)
    Y0 = (W0 * theta).sum(axis=1)
    a_n = min(p0*(1-p0), p1*(1-p1))

    b = 0

    for x in [0,1]:
        mu = [(Y0*(X==x)).sum() / (X==x).sum(), (Y1*(X==x)).sum() / (X==x).sum()]
        b += ((Y1-mu[1]-Y0+mu[0])**2*(X==x)).mean()

    return a_n*b


