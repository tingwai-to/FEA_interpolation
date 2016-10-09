from __future__ import print_function
from numpy import sqrt
from numba import jit
import numba as nb


@jit(nb.float64[:](nb.float64[:], nb.float64[:,:]), nopython=True)
def distance(xn, x):
    """Distance of nodes to points"""
    return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2)


@jit(nb.float64[:](nb.float64[:], nb.float64[:,:], nb.int32), nopython=True)
def weight(xn, x, power):
    """Weight of nodes to point"""
    return 1 / distance(xn, x)**power


@jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:],
     nb.float64, nb.float64, nb.float64, nb.int32),
     nopython=True)
def simple(p1, p2, p3, point, v1, v2, v3, power):
    """Simple inverse distance weighting function"""
    v_point = (v1*weight(p1, point, power) +
               v2*weight(p2, point, power) +
               v3*weight(p3, point, power)) / \
              (weight(p1, point, power) +
               weight(p2, point, power) +
               weight(p3, point, power))
    return v_point


def simple_nojit(p1, p2, p3, point, v1, v2, v3, power):
    """Non-JIT simple inverse distance weighting"""
    def distance(xn, x):
        return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2)
    def weight(xn, x, p):
        return 1 / distance(xn,x)**p

    v_point = (v1*weight(p1, point, power) +
               v2*weight(p2, point, power) +
               v3*weight(p3, point, power)) / \
              (weight(p1, point, power) +
               weight(p2, point, power) +
               weight(p3, point, power))
    return v_point
