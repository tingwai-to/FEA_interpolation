import time
import os
import sys
import interpolate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

# 2D sample points
p1 = np.array([2,2], dtype="f8")
p2 = np.array([4,3], dtype="f8")
p3 = np.array([1,4], dtype="f8")
# p4 = np.array([2,2])  # Value to be interpolated
v1 = 1
v2 = 2
v3 = 3

x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))

for N in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    x, y = np.mgrid[x_min:x_max:1j*N,
                    y_min:y_max:1j*N]
    t1 = time.time()
    #buff, mask = interpolate.interpolate2d(p1,p2,p3,
    buff = interpolate.interpolate2d(p1,p2,p3,
            np.array([_.ravel() for _ in (x, y)], dtype="f8"), v1, v2, v3)
    buff.shape = x.shape
    #mask.shape = x.shape
    t2 = time.time()
    buff = interpolate.interpolate2d_nojit(p1,p2,p3,
            np.array([_.ravel() for _ in (x, y)], dtype="f8"), v1, v2, v3)
    t3 = time.time()

    print ("% 5i Took %0.3e for JIT versus %0.3e for no-JIT (%0.2f x)" % (N,
            t2-t1, t3-t2, (t3-t2)/(t2-t1)))


# buffer = np.ma.MaskedArray(buff, ~mask).transpose()
# plt.clf()
# plt.imshow(buffer, origin='lower', interpolation='nearest')
# plt.colorbar()
# plt.savefig("output2d.png")
