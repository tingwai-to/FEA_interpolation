import numpy as np
import numpy.linalg as npla
from numba import cuda


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
result = np.empty((N**2), dtype='f8')

print (trans)
print (result, result.shape)
print (point, point.shape)

ref_point_x = trans[0,0]*(point[0]-p1[0]) + trans[0,1]*(point[1]-p1[0])
ref_point_y = trans[1,0]*(point[0]-p1[1]) + trans[1,1]*(point[1]-p1[1])

@cuda.jit
def increment_by_one(result, p1, p2, p3, v1, v2, v3, ref_point_x, ref_point_y):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos < result.size:  # Check array boundaries
        result[pos] = -1

result = np.empty((N**2), dtype='f8')

threadsperblock = 32
blockspergrid = (result.size + (threadsperblock - 1))
increment_by_one[blockspergrid, threadsperblock](
    result, p1, p2, p3, v1, v2, v3, ref_point_x, ref_point_y)

print (result)