from __future__ import print_function
from __future__ import division
import numpy as np
import numpy.linalg as npla
from numpy import sqrt
import numba as nb
from numba import jit
from numba import cuda
import matplotlib.pyplot as plt
from time import time


# CUDA and JIT shared kernel
def linear_kernel(result, p1, point, v1, v2, v3, trans):
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

cudacompiled = cuda.jit(nb.float64(nb.float64[:],
                                   nb.float64[:], nb.float64[:],
                                   nb.float64, nb.float64, nb.float64,
                                   nb.float64[:,:]),
                        device=True)(linear_kernel)
jitcompiled = jit(nb.float64(nb.float64[:],
                             nb.float64[:], nb.float64[:],
                             nb.float64, nb.float64, nb.float64,
                             nb.float64[:,:]),
                  nopython=True)(linear_kernel)


@jit(nb.void(nb.float64[:],
             nb.float64[:], nb.float64[:], nb.float64[:],
             nb.float64[:,:],
             nb.float64, nb.float64, nb.float64),
     nopython=True)
def linear_cpu_setup(result, p1, p2, p3, point, v1, v2, v3):
    trans = np.empty((2,2), dtype=point.dtype)
    trans[0,0] = p2[0] - p1[0]
    trans[0,1] = p3[0] - p1[0]
    trans[1,0] = p2[1] - p1[1]
    trans[1,1] = p3[1] - p1[1]
    trans = npla.inv(trans)

    for i in range(point.shape[0]):
        result[i] = jitcompiled(result, p1, point[i,:], v1, v2, v3, trans)

@cuda.jit(argtypes = [nb.float64[:],
                      nb.float64[:], nb.float64[:], nb.float64[:],
                      nb.float64[:,:],
                      nb.float64, nb.float64, nb.float64])
def linear_gpu_setup(result, p1, p2, p3, point, v1, v2, v3):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    # TODO: implement matrix inversion another way
    trans = cuda.shared.array(shape=(2,2), dtype=nb.float64)
    a = p2[0] - p1[0]
    b = p3[0] - p1[0]
    c = p2[1] - p1[1]
    d = p3[1] - p1[1]
    trans[0,0] = d/(a*d-b*c)
    trans[0,1] = -b/(a*d-b*c)
    trans[1,0] = -c/(a*d-b*c)
    trans[1,1] = a/(a*d-b*c)

    if pos < result.size:
        result[pos] = cudacompiled(result, p1, point[pos,:], v1, v2, v3, trans)


# Sample element
p1 = np.array([2, 2], dtype='f8')
p2 = np.array([4, 3], dtype='f8')
p3 = np.array([1, 4], dtype='f8')
v1 = 1.
v2 = 2.
v3 = 3.

x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))

N = 1024
x, y = np.mgrid[x_min:x_max:1j*N,
                y_min:y_max:1j*N]
point = np.array([_.ravel() for _ in (x, y)], dtype='f8').T


# CPU
cpu_result = np.empty((N**2), dtype='f8')
start = time()
for i in range(100):
    linear_cpu_setup(cpu_result, p1, p2, p3, point, v1, v2, v3)
end = time()
print(end-start)
cpu_result.shape = x.shape

# GPU
gpu_result = cuda.device_array((N**2), dtype='f8')
threadsperblock = 32
blockspergrid = (gpu_result.size + (threadsperblock - 1)) // threadsperblock
start = time()
for i in range(100):
    linear_gpu_setup[blockspergrid, threadsperblock](gpu_result, p1, p2, p3, point, v1, v2, v3)
end = time()
print(end-start)
gpu_result = gpu_result.copy_to_host()
gpu_result.shape = x.shape

diff = cpu_result.T - gpu_result.T

plt.figure()
plt.imshow(diff, extent=[x_min, x_max, y_min, y_max], origin='lower',
           interpolation='nearest')
plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.colorbar()
plt.show()