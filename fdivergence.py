import numpy as np
import tensorflow as tf
from scipy.special import digamma, gamma

""" Tensorflow implementation of f-divergence
Author: Suyong Choi (Department of Physics, Korea University)

Implements the calculations in the paper
'Nearest Neighbor Density Functional Estimation from Inverse Laplace Transform'
arXiv://1805.08342
"""

def knn_dist (A, B, k):  
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
    A,    [m,d] matrix
    B,    [n,d] matrix
    Returns:
    D,    [m,n] matrix of pairwise distances
    """
    # For small sample size, the following is OK. But for large sample size, requires too much memory
    #D=tf.reduce_sum((tf.expand_dims(A, 1)-tf.expand_dims(B, 0))**2,2)

    # memory efficient calculation
    # D[i, j] = A[i, l]*A[i,l] + B[j,m]*B[j,m] - 2A[i, k] B[j, k]
    A2 = tf.reshape(tf.reduce_sum(A*A, axis=1), [-1,1]) # -1 means infer the shape
    B2 = tf.reshape(tf.reduce_sum(B*B, axis=1), [1,-1])
    twoAB = 2*tf.matmul(A, tf.transpose(B))
    D = A2 + B2 - twoAB 

    #Dcorr = tf.where(tf.less(D, 0.0), tf.zeros_like(D), D) # if less than 0 then set it to 0
    Dcorr = tf.where(tf.less(D, 0.0), -D, D) # if less than 0 then set it to 0

    _, indices = tf.math.top_k(tf.negative(Dcorr), k)
    #print(indices)
    Dk = tf.gather(Dcorr, indices[:,k-1], axis=1, batch_dims=1)
    return Dk

def lebesgue_sphere_measure(n, r):
    """
    Calculates the Lebesgue measure of an n-dimensional sphere of radius r.

    Args:
        n (int): The dimension of the sphere.
        r (float): The radius of the sphere.

    Returns:
        float: The Lebesgue measure of the inside of the sphere.
    """
    pin2 = tf.math.pow(np.pi,  tf.cast(n/2, tf.float32))
    gamma = tf.cast(tf.math.exp(tf.math.lgamma(n/2 + 1)), tf.float32)
    res = pin2 / gamma * tf.math.pow(r, tf.cast(n, tf.float32))
    return res


def Ukm(xprobe, xdata, k, m):
    nd = tf.shape(xdata)[1] # dimension
    rk = tf.sqrt(knn_dist(xprobe, xdata, k))
    res = tf.cast(m, tf.float32) * lebesgue_sphere_measure(nd, rk)
    return res

# To estimate PDF
def pest(xprobe, xdata, k):
    m = tf.shape(xdata)[0] # data points
    nd = tf.shape(xdata)[1] # dimension
    rk = knn_dist(xprobe, xdata, k)
    res = k/(tf.cast(m, tf.float32) * lebesgue_sphere_measure(nd, rk))
    return res

# for single
def T_KL(xp, xdata, k):
    m = tf.shape(xp)[0] # data points
    ukm = Ukm(xp, xdata, k, m)
    return tf.reduce_mean(tf.math.log(ukm))

# KL divergence calculation using
# k Nearest Neighbor
# poor performance with peaks and cut off distributions
def D_KL(xp, xq, k, l, ppartofq = False):
    m = tf.shape(xp)[0] # p data points
    n = tf.shape(xq)[0] # q data points
    U = Ukm(xp, xp, k+1, m-1) # get  k+1 neightbor to exclude self
    if ppartofq:
        V = Ukm(xp, xq, l+1, n-1)
    else:
        V = Ukm(xp, xq, l, n)
    res = tf.reduce_mean(tf.math.log(V/U)) + digamma(k) - digamma(l) # D_KL(p||q)
    return res
    
# works better than D_JS, but still not good
# reproduction of distributions
def D_JS_fromDKL(xp, xq, k, l):
    mixpq = tf.concat([xp, xq], axis=0)
    res = 0.5 * D_KL(xp, mixpq, k, l, True)
    res += 0.5* D_KL(xq, mixpq, k, l, True)
    return res

# KL divergence calculation using
# k Nearest Neighbor
# symmetrized
# Shows best performance among the f-divergences
def D_KLsym(xp, xq, k, l):
    m = tf.shape(xp)[0] # p data points
    n = tf.shape(xq)[0] # q data points
    U = Ukm(xp, xp, k+1, m-1) # get  k+1 neightbor to exclude self
    V = Ukm(xp, xq, l, n)
    W = Ukm(xq, xq, l+1, n-1) # get  k+1 neightbor to exclude self
    Y = Ukm(xq, xp, k, m)
    res = tf.reduce_mean(tf.math.log(V/U)) + digamma(k) - digamma(l) # D_KL(p||q)
    res += tf.reduce_mean(tf.math.log(Y/W)) + digamma(l) - digamma(k)# D_KL(q||p)
    return res
    
def logcomb(n, r):
    """logarithm of combination nCr

    Args:
        n (integer): trials
        r (integer): number of successes

    Returns:
        float32: log[n!/r!/(n-r)!]
    """
    res = tf.math.log(gamma(n+1)) - tf.math.log(gamma(r+1)) - tf.math.log(gamma(n-r+1))
    res = tf.cast(res, tf.float32)
    return res

def B_KL_lt(k, l, u, v):
    c  = tf.math.exp(-logcomb(k+l-2, k-1))
    sumlt = tf.zeros_like(u, dtype=tf.float32)
    for j in range(1, l):
        sumlt += tf.math.exp(logcomb(k+l-2, k-1+j)) * tf.math.pow( -u/v, j)/j
    sumlt = tf.cast(c*sumlt, tf.float32)

    return sumlt

def B_KL_ge(k, l, u, v):
    c  = tf.math.exp(-logcomb(k+l-2, k-1))
    sumge = tf.zeros_like(u, dtype=tf.float32)
    for j in range(-k+1, 0):
        sumge += -1.0*tf.math.exp(logcomb(k+l-2, k-1+j)) * tf.math.pow( -u/v, j)/j

    for j in range(-k+1, 0):
        sumge += -1.0*tf.math.exp(logcomb(k+l-2, k-1+j)) * tf.math.pow( -1.0, j)/j

    for j in range(1, l):
        sumge += -1.0*tf.math.exp(logcomb(k+l-2, k-1+j)) * tf.math.pow( -1.0, j)/j

    sumge = c*sumge - tf.math.log(u/v)

    return sumge



def B_KL(k, l, u, v):
    sumlt = B_KL_lt(k, l, u, v)
    sumge = B_KL_ge(k, l, u, v)
    res = tf.where(u<v, sumlt, sumge)

    return res

# really poor performance, peaks and what not
def D_JS(xp, xq, k, l):
    """ JS-divergence in example IV.5

    Args:
        xp (tensor): _description_
        xq (tensor): _description_
        k (int): _description_
        l (int): _description_

    Returns:
        _type_: _description_
    """
    m = tf.shape(xp)[0] # p data points
    n = tf.shape(xq)[0] # q data points
    U = Ukm(xp, xp, k+1, m-1) # get  k+1 neightbor to exclude self
    V = Ukm(xp, xq, l, n)
    res = 0.5* (tf.math.log(2.0) + (l-1)/k *U/V * (digamma(l-1) - digamma(k+1) + tf.math.log(2*U/V))
                +B_KL(k, l, U, V) + (l-1)/k * U/V* B_KL(k+1, l-1, U, V))
    res = tf.reduce_mean(res)
    return res

# alpha divergence
def alphadiv(xp, xq, k, l, alpha):
    assert k>(alpha-1.0) and l>(1.0-alpha)
    m = tf.shape(xp)[0] # p data points
    n = tf.shape(xq)[0] # q data points
    U = Ukm(xp, xp, k+1, m-1) # get  k+1 neightbor to exclude self
    V = Ukm(xp, xq, l, n)
    coef= gamma(k)*gamma(l) / gamma(k-alpha+1) / gamma(l+alpha-1.0)
    res = coef * tf.reduce_mean(tf.math.pow(V/U, alpha-1.0)) 
    return res

# symmetric version of alpha divergence
def alphadivsym(xp, xq, k, l, alpha):
    res_pq = alphadiv(xp, xq, k, l, alpha)
    res_qp = alphadiv(xq, xp, k, l, alpha)
    return (res_pq + res_qp)/2