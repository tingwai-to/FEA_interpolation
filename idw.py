from __future__ import print_function
import numpy as np
from numpy import sqrt
from numba import jit
import numba as nb


def make_2d_idw_jit(dtype):
    @jit(dtype[:](dtype[:], dtype[:,:]), nopython=True)
    def distance(xn, x):
        """Distance of nodes to points"""
        return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2)

    @jit(dtype[:](dtype[:], dtype[:,:], nb.int32), nopython=True)
    def weight(xn, x, power):
        """Weight of nodes to point"""
        return 1 / distance(xn, x)**power

    @jit(dtype[:](dtype[:], dtype[:], dtype[:], dtype[:,:],
         dtype, dtype, dtype, nb.int32),
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
    return simple


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


idw_2d_f64 = make_2d_idw_jit(nb.float64)
idw_2d_f32 = make_2d_idw_jit(nb.float32)
