import time
import os
import sys
import interpolate
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# matplotlib.use("agg")

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


fig, ax = plt.subplots()

for N in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    x, y = np.mgrid[x_min:x_max:1j*N,
                    y_min:y_max:1j*N]
    start = time.time()
    buff = interpolate.interpolate2d_f64(p1,p2,p3,
            np.array([_.ravel() for _ in (x, y)], dtype="f8"), v1, v2, v3)
    end = time.time()
    ax.scatter(N, end-start, c='g', s=50)

    start = time.time()
    buff = interpolate.interpolate2d_f32(p1.astype(np.float32), p2.astype(np.float32), p3.astype(np.float32),
            np.array([_.ravel() for _ in (x, y)], dtype="f"), v1, v2, v3)
    end = time.time()
    ax.scatter(N, end-start, c='r', s=50)

    start = time.time()
    buff = interpolate.interpolate2d_nojit(p1,p2,p3,
            np.array([_.ravel() for _ in (x, y)], dtype="f8"), v1, v2, v3)
    end = time.time()
    ax.scatter (N, end-start, c='b', s=50)

ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
f64_patch = mpatches.Patch(color='green', label='float64')
f32_patch = mpatches.Patch(color='red', label='float32')
nojit_patch = mpatches.Patch(color='blue', label='nojit64')
plt.legend(handles=[f64_patch, f32_patch, nojit_patch], loc=2)
plt.savefig("2d_speed_comparison.png")

# buffer = np.ma.MaskedArray(buff, ~mask).transpose()
# plt.clf()
# plt.imshow(buffer, origin='lower', interpolation='nearest')
# plt.colorbar()
# plt.savefig("output2d.png")
