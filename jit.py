from __future__ import print_function
from __future__ import division
import numpy as np
import numba as nb
from numba import jit
from numba import cuda
import matplotlib.pyplot as plt
from time import time
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
    p = np.empty(2, dtype=point.dtype)
    for i in range(point.shape[0]):
        p[0] = point[i, 0]
        p[1] = point[i, 1]
        result[i] = jit_kernel.nearest_jit(p1, p2, p3, p, v1, v2, v3)

@cuda.jit(nb.void(nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:,:],
                  nb.float64, nb.float64, nb.float64))
def nearest_gpu_setup(result, p1, p2, p3, point, v1, v2, v3):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    p = cuda.local.array(2, nb.float64)

    if pos < result.size:
        p[0] = point[pos,0]
        p[1] = point[pos,1]
        result[pos] = jit_kernel.nearest_cuda(p1, p2, p3, p, v1, v2, v3)


from sample_2d_points import *

# CPU
cpu_result = np.empty((N**2), dtype='f8')
start = time()
for i in range(1):
    idw_cpu_setup(cpu_result, p1, p2, p3, points_f64, v1, v2, v3, power)
end = time()
print('{:20s} {:f}\n'.format('CPU 64', end-start))

cpu_result_32 = np.empty((N**2), dtype='f')
start = time()
for i in range(1):
    idw_cpu_setup(cpu_result_32, p1_f32, p2_f32, p3_f32, points_f32, v1, v2, v3, power)
end = time()
print('{:20s} {:f}\n'.format('CPU 32', end-start))


# GPU
gpu_result = cuda.device_array((N**2), dtype='f8')
threadsperblock = 32
blockspergrid = (gpu_result.size + (threadsperblock - 1)) // threadsperblock

# Copy array from host -> device
start = time()
p1 = cuda.to_device(p1)
p2 = cuda.to_device(p2)
p3 = cuda.to_device(p3)
point = cuda.to_device(points_f64)
trans = cuda.to_device(trans)
end = time()
print('{:20s} {:f}'.format('Copy to device', end-start))

start = time()
with cuda.profiling():
    for i in range(1):
        idw_gpu_setup[blockspergrid, threadsperblock]\
            (gpu_result, p1, p2, p3, point, v1, v2, v3, power)
end = time()
print('{:20s} {:f}'.format('GPU 64', end-start))

# Copy array back, device -> host
start = time()
gpu_result = gpu_result.copy_to_host()
end =time()
print('{:20s} {:f}'.format('Copy back to host', end-start))


cpu_result.shape = x.shape
cpu_result_32.shape = x.shape
gpu_result.shape = x.shape
diff = cpu_result.T - gpu_result.T

plt.figure()
plt.imshow(cpu_result.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
           interpolation='nearest')
plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.colorbar()
plt.show()
