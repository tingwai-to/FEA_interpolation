from node import Node2D
from element import Elem2D
import element
import linear
import idw
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from time import time
import os
import sys


# 2D sample points
p1 = np.array([2,2], dtype='f8')
p2 = np.array([4,3], dtype='f8')
p3 = np.array([1,4], dtype='f8')
v1 = 1.
v2 = 2.
v3 = 3.

node1 = Node2D(p1, v1)
node2 = Node2D(p2, v2)
node3 = Node2D(p3, v3)
triangle = Elem2D(node1, node2, node3)

x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))


# JIT vs non-JIT
fig, ax = plt.subplots()
jit64 = []
jit32 = []
nojit32 = []
nojit64 = []
idw64 = []
idw32 = []
nojitidw64 = []
nojitidw32 = []
Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

for N in Ns:
    x, y = np.mgrid[x_min:x_max:1j*N,
                    y_min:y_max:1j*N]

    points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
    points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')

    # IDW JIT
    start = time()
    triangle.sample("idw", points_f64, jit=True, power=2)
    end = time()
    idw64.append(end-start)

    start = time()
    triangle.sample("idw", points_f32, jit=True, power=2)
    end = time()
    idw32.append(end-start)

    # IDW NO-JIT
    start = time()
    triangle.sample('idw', points_f64, power=2)
    end = time()
    nojitidw64.append(end-start)

    start = time()
    triangle.sample('idw', points_f32, power=2)
    end = time()
    nojitidw32.append(end-start)

    # Linear JIT
    start = time()
    triangle.sample("linear", points_f64, jit=True)
    end = time()
    jit64.append(end-start)

    start = time()
    triangle.sample("linear", points_f32, jit=True)
    end = time()
    jit32.append(end-start)

    # Linear NO-JIT
    start = time()
    triangle.sample('linear', points_f64)
    end = time()
    nojit64.append(end-start)

    start = time()
    triangle.sample('linear', points_f32)
    end = time()
    nojit32.append(end-start)


ax.loglog(Ns, jit64, '-og', label='JIT-64')
ax.loglog(Ns, jit32, '-or', label='JIT-32')
ax.loglog(Ns, nojit64, '-ob', label='NOJIT-64')
ax.loglog(Ns, nojit32, '-oy', label='NOJIT-32')
ax.loglog(Ns, idw64, '-om', label='IDW-64')
ax.loglog(Ns, idw32, '-o', color='#A9A9A9', label='IDW-32')
ax.loglog(Ns, nojitidw64, '-ok', label='NOJIT-IDW-64')
ax.loglog(Ns, nojitidw32, '-oc', label='NOJIT-IDW-32')
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
idw64 = np.array(idw64)
idw32 = np.array(idw32)
nojitidw64 = np.array(nojitidw64)
nojitidw32 = np.array(nojitidw32)

rel_jit64 = (jit64 / (Ns*Ns))
rel_jit32 = (jit32 / (Ns*Ns))
rel_nojit64 = (nojit64 / (Ns*Ns))
rel_nojit32 = (nojit32 / (Ns*Ns))
rel_idw64 = (idw64 / (Ns*Ns))
rel_idw32 = (idw32 / (Ns*Ns))
rel_nojitidw64 = (nojitidw64 / (Ns*Ns))
rel_nojitidw32 = (nojitidw32 / (Ns*Ns))

plt.clf()
ax = plt.gca()
ax.loglog(Ns, rel_jit64, '-og', label='JIT-64')
ax.loglog(Ns, rel_jit32, '-or', label='JIT-32')
ax.loglog(Ns, rel_nojit64, '-ob', label='NOJIT-64')
ax.loglog(Ns, rel_nojit32, '-oy', label='NOJIT-32')
ax.loglog(Ns, rel_idw64, '-om', label='IDW-64')
ax.loglog(Ns, rel_idw32, '-o', color='#A9A9A9', label='IDW-32')
ax.loglog(Ns, rel_nojitidw64, '-ok', label='NOJIT-IDW-64')
ax.loglog(Ns, rel_nojitidw32, '-oc', label='NOJIT-IDW-32')
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
plt.legend(loc=1)
plt.xlabel('N')
plt.ylabel('Time per Element')
plt.savefig("2d_speed_comparison_rel.png")


"""
# Plot individual function
N=128
x, y = np.mgrid[x_min:x_max:1j*N,
                y_min:y_max:1j*N]

points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')

power = 128
buff = element.linear_2d_jit(triangle, points_f64)
buff.shape = x.shape


plt.imshow(buff.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
           interpolation='nearest')
plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.colorbar()
plt.savefig('output_2d.png')
"""
