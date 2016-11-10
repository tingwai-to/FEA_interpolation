from __future__ import division
import numpy as np
import numpy.linalg as npla
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
from time import time


@cuda.jit(argtypes = [nb.float64[:],
                      nb.float64[:],
                      nb.float64[:,:],
                      nb.float64, nb.float64, nb.float64,
                      nb.float64[:,:]])
def linear_2d_cuda(result, p1, point, v1, v2, v3, trans):
    for k in range(1):
        # Thread id in a 1D block
        tx = cuda.threadIdx.x
        # Block id in a 1D grid
        ty = cuda.blockIdx.x
        # Block width, i.e. number of threads per block
        bw = cuda.blockDim.x
        # Compute flattened index inside the array
        pos = tx + ty * bw

        if pos < result.size:  # Check array boundaries
            ref_point_x = trans[0,0]*(point[0,pos]-p1[0]) + \
                          trans[0,1]*(point[1,pos]-p1[0])
            ref_point_y = trans[1,0]*(point[0,pos]-p1[1]) + \
                          trans[1,1]*(point[1,pos]-p1[1])
            area2 = 0.5*ref_point_x
            area3 = 0.5*ref_point_y
            area1 = 0.5 - area2 - area3

            # TODO: implement masking without if statements
            #if (area1/0.5) < 0 or (area1/0.5) > 1 or \
            #   (area2/0.5) < 0 or (area2/0.5) > 1 or \
            #   (area3/0.5) < 0 or (area3/0.5) > 1:
            #    result[pos] = -1
            #else:
            result[pos] = v1*(area1/0.5) + v2*(area2/0.5) + v3*(area3/0.5)


@cuda.jit(argtypes = [nb.float64[:],
                      nb.float64[:],
                      nb.float64[:,:],
                      nb.float64, nb.float64, nb.float64, nb.float64,
                      nb.float64[:,:]])
def linear_3d_cuda(result, p1, point, v1, v2, v3, v4, trans):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos < result.size:  # Check array boundaries
        ref_point_x = trans[0,0]*(point[0,pos]-p1[0]) + \
                      trans[0,1]*(point[1,pos]-p1[0]) + \
                      trans[0,2]*(point[2,pos]-p1[0])
        ref_point_y = trans[1,0]*(point[0,pos]-p1[1]) + \
                      trans[1,1]*(point[1,pos]-p1[1]) + \
                      trans[1,2]*(point[2,pos]-p1[1])
        ref_point_z = trans[2,0]*(point[0,pos]-p1[2]) + \
                      trans[2,1]*(point[1,pos]-p1[2]) + \
                      trans[2,2]*(point[2,pos]-p1[2])
        vol2 = 1./6*ref_point_x
        vol3 = 1./6*ref_point_y
        vol4 = 1./6*ref_point_z
        vol1 = 1./6 - vol2 - vol3 - vol4

        # if vol1/(1./6) < 0 or \
        #    vol1/(1./6) > 1 or \
        #    vol2/(1./6) > 0 or \
        #    vol2/(1./6) > 1 or \
        #    vol3/(1./6) > 0 or \
        #    vol3/(1./6) > 1:
        #     result[pos] = -1
        # else:
        result[pos] = v1*(vol1/(1./6)) + v2*(vol2/(1./6)) + \
                      v3*(vol3/(1./6)) + v4*(vol4/(1./6))


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

trans = np.empty((2,2), dtype='f8')
trans[0,0] = p2[0] - p1[0]
trans[0,1] = p3[0] - p1[0]
trans[1,0] = p2[1] - p1[1]
trans[1,1] = p3[1] - p1[1]
trans = npla.inv(trans)

point = np.array([_.ravel() for _ in (x, y)], dtype='f8')
point = point.reshape((N,N,2))


## CUDA setup
# Allocate empty device ndarray
d_result = cuda.device_array((N**2), dtype='f8')

threadsperblock = 32
blockspergrid = (d_result.size + (threadsperblock - 1)) // threadsperblock
print (blockspergrid, threadsperblock)
start = time()
linear_2d_cuda[blockspergrid, threadsperblock](d_result,
                                               p1, point, v1, v2, v3, trans)
end = time()
print(end-start)

# Copy device->host to an existing array
h_result = d_result.copy_to_host()
h_result.shape = x.shape

plt.figure()
plt.imshow(h_result.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
           interpolation='nearest')
plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.colorbar()
plt.show()