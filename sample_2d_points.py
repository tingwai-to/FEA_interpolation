import time
import os
import sys
import interpolate
import idw
import numpy as np
import matplotlib.pyplot as plt

# 2D sample points
p1 = np.array([2,2], dtype='f8')
p2 = np.array([4,3], dtype='f8')
p3 = np.array([1,4], dtype='f8')
v1 = 1.
v2 = 2.
v3 = 3.

x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))


# Plot using inverse distance weighting
N=128
x, y = np.mgrid[x_min:x_max:1j*N,
                y_min:y_max:1j*N]

points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')

power = 128
buff = idw.simple(p1, p2, p3, points_f64, v1, v2, v3, power)
buff.shape = x.shape


plt.imshow(buff.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
           interpolation='nearest')
plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.colorbar()
plt.savefig('idw_2d.png')


# Linear interpolation, JIT & non-JIT
fig, ax = plt.subplots()
jit64 = []
jit32 = []
nojit32 = []
nojit64 = []
Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

for N in Ns:
    x, y = np.mgrid[x_min:x_max:1j*N,
                    y_min:y_max:1j*N]

    points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
    points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')

    start = time.time()
    buff = interpolate.linear_2d_f64(p1, p2, p3, points_f64,
                                     v1, v2, v3)
    end = time.time()
    jit64.append(end-start)

    start = time.time()
    buff = interpolate.linear_2d_f32(p1.astype(np.float32),
                                     p2.astype(np.float32),
                                     p3.astype(np.float32),
                                     points_f32,
                                     v1, v2, v3)
    end = time.time()
    jit32.append(end-start)

    start = time.time()
    buff = interpolate.linear_2d_nojit64(p1, p2, p3, points_f64,
                                         v1, v2, v3)
    end = time.time()
    nojit64.append(end-start)

    start = time.time()
    buff = interpolate.linear_2d_nojit32(p1.astype(np.float32),
                                         p2.astype(np.float32),
                                         p3.astype(np.float32),
                                         points_f32,
                                         v1, v2, v3)
    end = time.time()
    nojit32.append(end-start)


ax.loglog(Ns, jit64, '-og', label='JIT-64')
ax.loglog(Ns, jit32, '-or', label='JIT-32')
ax.loglog(Ns, nojit64, '-ob', label='NOJIT-64')
ax.loglog(Ns, nojit32, '-oy', label='NOJIT-32')
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
plt.legend(loc=2)
plt.xlabel('N')
plt.ylabel('Time [s]')
plt.savefig("2d_speed_comparison.png")


jit64 = np.array(jit64)
jit32 = np.array(jit32)
nojit32 = np.array(nojit32)
nojit64 = np.array(nojit64)

rel_jit64 = (jit64 / (Ns*Ns))
rel_jit32 = (jit32 / (Ns*Ns))
rel_nojit64 = (nojit64 / (Ns*Ns))
rel_nojit32 = (nojit32 / (Ns*Ns))

plt.clf()
ax = plt.gca()
ax.loglog(Ns, rel_jit64, '-og', label='JIT-64')
ax.loglog(Ns, rel_jit32, '-or', label='JIT-32')
ax.loglog(Ns, rel_nojit64, '-ob', label='NOJIT-64')
ax.loglog(Ns, rel_nojit32, '-oy', label='NOJIT-32')
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
plt.legend(loc=2)
plt.xlabel('N')
plt.ylabel('Time per Element')
plt.savefig("2d_speed_comparison_rel.png")
