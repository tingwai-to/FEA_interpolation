from __future__ import print_function
from __future__ import division
import numpy as np
import numba as nb
from numba import jit
from numba import cuda
import jit_kernel


@jit([nb.void(nb.float64[:],
              nb.float64[:], nb.float64[:], nb.float64[:],
              nb.float64[:,:],
              nb.float64, nb.float64, nb.float64,
              nb.float64[:,:]),
      nb.void(nb.float32[:],
              nb.float32[:], nb.float32[:], nb.float32[:],
              nb.float32[:,:],
              nb.float32, nb.float32, nb.float32,
              nb.float32[:,:])],
     nopython=True)
def linear_cpu_setup(result, p1, p2, p3, point, v1, v2, v3, trans):
    """Interpolate using CPU linear kernel

    Note:
        Returns nothing. Kernel populates result argument
    """
    p = np.empty(2, dtype=point.dtype)
    for i in range(point.shape[0]):
        p[0] = point[i,0]
        p[1] = point[i,1]
        result[i] = jit_kernel.linear_jit(p1, p, v1, v2, v3, trans)

@cuda.jit(nb.void(nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.float64[:,:]))
def linear_gpu_setup(result, p1, p2, p3, point, v1, v2, v3, trans):
    """Interpolate using GPU linear kernel

    Note:
        Returns nothing. Kernel populates result argument
    """
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    p = cuda.local.array(2, nb.float64)

    if pos < result.size:
        p[0] = point[pos,0]
        p[1] = point[pos,1]
        result[pos] = jit_kernel.linear_cuda(p1, p, v1, v2, v3, trans)

@jit([nb.void(nb.float64[:],
              nb.float64[:], nb.float64[:], nb.float64[:],
              nb.float64[:,:],
              nb.float64, nb.float64, nb.float64,
              nb.int64),
      nb.void(nb.float32[:],
              nb.float32[:], nb.float32[:], nb.float32[:],
              nb.float32[:,:],
              nb.float32, nb.float32, nb.float32,
              nb.int32)],
     nopython=True)
def idw_cpu_setup(result, p1, p2, p3, point, v1, v2, v3, power):
    """Interpolate using CPU inverse distance weighting kernel

    Note:
        Returns nothing. Kernel populates result argument
    """
    p = np.empty(2, dtype=point.dtype)
    for i in range(point.shape[0]):
        p[0] = point[i,0]
        p[1] = point[i,1]
        result[i] = jit_kernel.idw_jit(p1, p2, p3, p, v1, v2, v3, power)

@cuda.jit(nb.void(nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64,
                  nb.int64))
def idw_gpu_setup(result, p1, p2, p3, point, v1, v2, v3, power):
    """Interpolate using GPU inverse distance weighting kernel

    Note:
        Returns nothing. Kernel populates result argument
    """
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    p = cuda.local.array(2, nb.float64)

    if pos < result.size:
        p[0] = point[pos,0]
        p[1] = point[pos,1]
        result[pos] = jit_kernel.idw_cuda(p1, p2, p3, p, v1, v2, v3, power)

@jit([nb.void(nb.float64[:],
              nb.float64[:], nb.float64[:], nb.float64[:],
              nb.float64[:,:],
              nb.float64, nb.float64, nb.float64),
      nb.void(nb.float32[:],
              nb.float32[:], nb.float32[:], nb.float32[:],
              nb.float32[:,:],
              nb.float32, nb.float32, nb.float32)],
     nopython=True)
def nearest_cpu_setup(result, p1, p2, p3, point, v1, v2, v3):
    """Interpolate using CPU nearest neighbor kernel

    Note:
        Returns nothing. Kernel populates result argument
    """
    p = np.empty(2, dtype=point.dtype)
    for i in range(point.shape[0]):
        p[0] = point[i,0]
        p[1] = point[i,1]
        result[i] = jit_kernel.nearest_jit(p1, p2, p3, p, v1, v2, v3)

@cuda.jit(nb.void(nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64))
def nearest_gpu_setup(result, p1, p2, p3, point, v1, v2, v3):
    """Interpolate using GPU nearest neighbor kernel

    Note:
        Returns nothing. Kernel populates result argument
    """
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    p = cuda.local.array(2, nb.float64)

    if pos < result.size:
        p[0] = point[pos,0]
        p[1] = point[pos,1]
        result[pos] = jit_kernel.nearest_cuda(p1, p2, p3, p, v1, v2, v3)
