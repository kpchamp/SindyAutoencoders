import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library(X, poly_order, include_sine=False):
    m,n = X.shape
    l = library_size(n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            index += 1

    return library


def sindy_library_order2(X, dX, poly_order, include_sine=False):
    m,n = X.shape
    l = library_size(2*n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    X_combined = np.concatenate((X, dX), axis=1)

    for i in range(2*n):
        library[:,index] = X_combined[:,i]
        index += 1

    if poly_order > 1:
        for i in range(2*n):
            for j in range(i,2*n):
                library[:,index] = X_combined[:,i]*X_combined[:,j]
                index += 1

    if poly_order > 2:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        for r in range(q,2*n):
                            library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]*X_combined[:,r]
                            index += 1

    if include_sine:
        for i in range(2*n):
            library[:,index] = np.sin(X_combined[:,i])
            index += 1

    return library


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    m = t.size
    n = 2*x0.size
    l = Xi.shape[0]

    Xi_order1 = np.zeros((l,n))
    for i in range(n//2):
        Xi_order1[2*(i+1),i] = 1.
        Xi_order1[:,i+n//2] = Xi[:,i]
    
    x = sindy_simulate(np.concatenate((x0,dx0)), t, Xi_order1, poly_order, include_sine)
    return x
