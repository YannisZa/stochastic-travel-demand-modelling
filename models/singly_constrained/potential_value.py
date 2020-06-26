import numpy as np

# As defined in C function
def _logsumexp(xx,m):
    _max = xx[0]
    for j in range(m):
        if xx[j] > _max:
            _max = xx[j]

    _sum = 0;
    for j in range(m):
        _sum += np.exp(xx[j] - _max);

    return _max + np.log(_sum)

def pot_value(x,o,c,params):
    pot = 1
    a = params[0]
    b = params[1]
    d = params[2]
    g = params[3]
    k = params[4]
    e = params[5]

    N,M = c.shape

    a_inv = 1./a

    grad = np.zeros(M)
    wksp = np.zeros(M)

    utility = 0
    for i in range(N):
        for j in range(M):
            wksp[j] = a*x[j] - b*c[i,j]
        temp = _logsumexp(wksp,M)

        utility += -e*a_inv*o[i]*temp;

        for j in range(M):
            grad[j] += -e*o[i]*np.exp(wksp[j]-temp)

    for j in range(M):
        grad[j] += k*np.exp(x[j]) - d
        grad[j] *= g


    # Compute cost and additional utilities
    cost = k * np.sum(np.exp(x))
    additional = -d * np.sum(x)

    potential = g*e*(utility+cost+additional)

    return potential,grad
