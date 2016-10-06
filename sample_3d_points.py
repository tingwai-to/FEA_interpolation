import time
import os
import sys
import interpolate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

# 3D sample points
p1 = np.array([2,2,1])
p2 = np.array([4,3.5,5])
p3 = np.array([1,4,3])
p4 = np.array([6,5,3])
#p5 = np.array([3,3.5,3]) # Value to be interpolated
v1 = 0.7
v2 = 2.2
v3 = 3.4
v4 = 4.6

x_min = min(_[0] for _ in (p1, p2, p3, p4))
y_min = min(_[1] for _ in (p1, p2, p3, p4))
z_min = min(_[2] for _ in (p1, p2, p3, p4))
x_max = max(_[0] for _ in (p1, p2, p3, p4))
y_max = max(_[1] for _ in (p1, p2, p3, p4))
z_max = max(_[2] for _ in (p1, p2, p3, p4))

x, y, z = np.mgrid[x_min:x_max:1j*N,
                   y_min:y_max:1j*N,
                   z_min:z_max:1j*N]

buff, mask = interpolate.interpolate3d(p1,p2,p3,p4,
        np.array([_.ravel() for _ in (x, y, z)], dtype="f8"),
        v1, v2, v3, v4)
buff.shape = x.shape
mask.shape = x.shape

if not os.path.isdir("frames3d"):
    os.mkdir("frames3d")

buffer = np.ma.MaskedArray(buff, ~mask).transpose()

for i in range(N):
    print(i)

    plt.clf()
    plt.imshow(buffer[i,:,:], origin='lower', interpolation='nearest',
            extent=[y_min, y_max, z_min, z_max])
    plt.clim(1, 4)
    plt.colorbar()
    plt.savefig("frames/slice_x_%03i.png" % i)

    plt.clf()
    plt.imshow(buffer[:,i,:], origin='lower', interpolation='nearest',
            extent=[x_min, x_max, z_min, z_max])
    plt.clim(1, 4)
    plt.colorbar()
    plt.savefig("frames/slice_y_%03i.png" % i)

    plt.clf()
    plt.imshow(buffer[:,:,i], origin='lower', interpolation='nearest',
            extent=[x_min, x_max, y_min, y_max])
    plt.clim(1, 4)
    plt.colorbar()
    plt.savefig("frames/slice_z_%03i.png" % i)
