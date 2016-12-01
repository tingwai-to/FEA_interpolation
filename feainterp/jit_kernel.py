from __future__ import print_function
from __future__ import division
from math import sqrt
import numba as nb
from numba import jit
from numba import cuda


def linear_kernel(p1, point, v1, v2, v3, trans):
    """CPU/GPU shared kernel for linear 2D interpolation"""
    ref_point_x = trans[0,0]*(point[0]-p1[0]) + \
                  trans[0,1]*(point[1]-p1[0])
    ref_point_y = trans[1,0]*(point[0]-p1[1]) + \
                  trans[1,1]*(point[1]-p1[1])
    area2 = 0.5*ref_point_x
    area3 = 0.5*ref_point_y
    area1 = 0.5 - area2 - area3

    if (area1/0.5) < 0 or \
       (area1/0.5) > 1 or \
       (area2/0.5) < 0 or \
       (area2/0.5) > 1 or \
       (area3/0.5) < 0 or \
       (area3/0.5) > 1:
        return -1
    else:
        return v1*(area1/0.5) + v2*(area2/0.5) + v3*(area3/0.5)

# Compiles linear kernel for CPU and CUDA
linear_cuda = cuda.jit(nb.float64(nb.float64[:], nb.float64[:],
                                  nb.float64, nb.float64, nb.float64,
                                  nb.float64[:,:]),
                       device=True)(linear_kernel)
linear_jit = jit([nb.float64(nb.float64[:], nb.float64[:],
                             nb.float64, nb.float64, nb.float64,
                             nb.float64[:,:]),
                  nb.float32(nb.float32[:], nb.float32[:],
                             nb.float32, nb.float32, nb.float32,
                             nb.float32[:,:])],
                 nopython=True)(linear_kernel)


def idw_kernel(p1, p2, p3, point, v1, v2, v3, power):
    """CPU/GPU shared kernel for inverse distance weighting 2D interpolation"""
    distance_p1 = sqrt((p1[0]-point[0])**2 + (p1[1]-point[1])**2)**power
    distance_p2 = sqrt((p2[0]-point[0])**2 + (p2[1]-point[1])**2)**power
    distance_p3 = sqrt((p3[0]-point[0])**2 + (p3[1]-point[1])**2)**power

    weights = 0
    v_weights = 0
    distance_p1 = max(distance_p1, 1e-16)
    distance_p2 = max(distance_p2, 1e-16)
    distance_p3 = max(distance_p3, 1e-16)
    weight_p1 = 1 / distance_p1
    v_weight_p1 = v1*weight_p1
    weights += weight_p1
    v_weights += v_weight_p1
    weight_p2 = 1 / distance_p2
    v_weight_p2 = v2*weight_p2
    weights += weight_p2
    v_weights += v_weight_p2
    weight_p3 = 1 / distance_p3
    v_weight_p3 = v3*weight_p3
    weights += weight_p3
    v_weights += v_weight_p3

    if v_weights == 0:
        return -1
    v_point = v_weights/weights
    return v_point

# Compiles inverse distance weighting kernel for CPU and CUDA
idw_cuda = cuda.jit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:],
                               nb.float64[:],
                               nb.float64, nb.float64, nb.float64,
                               nb.int64),
                    device=True)(idw_kernel)
idw_jit = jit([nb.float64(nb.float64[:], nb.float64[:], nb.float64[:],
                          nb.float64[:],
                          nb.float64, nb.float64, nb.float64,
                          nb.int64),
               nb.float32(nb.float32[:], nb.float32[:], nb.float32[:],
                          nb.float32[:],
                          nb.float32, nb.float32, nb.float32,
                          nb.int32)],
              nopython=True)(idw_kernel)


def nearest_kernel(p1, p2, p3, point, v1, v2, v3):
    """CPU/GPU shared kernel for nearest neighbor 2D interpolation"""
    distance_p1 = sqrt((p1[0]-point[0])**2 + (p1[1]-point[1])**2)
    distance_p2 = sqrt((p2[0]-point[0])**2 + (p2[1]-point[1])**2)
    distance_p3 = sqrt((p3[0]-point[0])**2 + (p3[1]-point[1])**2)

    distances = (distance_p1, distance_p2, distance_p3)

    if min(distances) == distance_p1:
        return v1
    elif min(distances) == distance_p2:
        return v2
    else:
        return v3

# Compiles nearest neighbor kernel for CPU and CUDA
nearest_cuda = cuda.jit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:],
                                   nb.float64[:],
                                   nb.float64, nb.float64, nb.float64),
                        device=True)(nearest_kernel)
nearest_jit = jit([nb.float64(nb.float64[:], nb.float64[:], nb.float64[:],
                              nb.float64[:],
                              nb.float64, nb.float64, nb.float64),
                   nb.float32(nb.float32[:], nb.float32[:], nb.float32[:],
                              nb.float32[:],
                              nb.float32, nb.float32, nb.float32)],
                  nopython=True)(nearest_kernel)
