from node import Node2D
from element import Elem2D
import element
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numba as nb
from time import time
import os
import sys

# 2D sample points
p1 = np.array([2, 2], dtype='f8')
p2 = np.array([4, 3], dtype='f8')
p3 = np.array([1, 4], dtype='f8')
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


def speed_comparison():
    # JIT vs non-JIT
    fig, ax = plt.subplots()
    funcs = [('linear', 64, True), ('linear', 32, True),
             ('linear', 64, False), ('linear', 32, False),
             ('idw', 64, True), ('idw', 32, True),
             ('idw', 64, False), ('idw', 32, False),
             ('nearest', 64, True), ('nearest', 32, True),
             ('nearest', 64, False), ('nearest', 32, False)]

    Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    times = np.empty((len(funcs), Ns.shape[0]))

    for i, N in enumerate(Ns):
        x, y = np.mgrid[x_min:x_max:1j*N,
                        y_min:y_max:1j*N]
        points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
        points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')

        for j, fun in enumerate(funcs):
            if fun[0] == 'idw':
                start = time()
                triangle.sample(fun[0], eval('points_f%s' % fun[1]), jit=fun[2], power=2)
                end = time()
            else:
                start = time()
                triangle.sample(fun[0], eval('points_f%s' % fun[1]), jit=fun[2])
                end = time()
            times[j, i] = end-start

    colors = cm.nipy_spectral(np.linspace(0, 1, len(funcs)))
    for c, row in enumerate(times):
        ax.loglog(Ns, row, color=colors[c], marker='o', ls='-',
                  label='%s-%d, jit=%r' % (funcs[c][0], funcs[c][1], funcs[c][2]))
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)
    plt.legend(loc=2, prop={'size':10})
    plt.xlabel('N')
    plt.ylabel('Time [s]')
    plt.savefig("2d_speed_comparison.png")

    # TODO: clean up relative speed
    jit64 = np.array(jit64)
    jit32 = np.array(jit32)
    nojit32 = np.array(nojit32)
    nojit64 = np.array(nojit64)
    idw64 = np.array(idw64)
    idw32 = np.array(idw32)
    nojitidw64 = np.array(nojitidw64)
    nojitidw32 = np.array(nojitidw32)
    nn64 = np.array(nn64)
    nn32 = np.array(nn32)
    nojitnn64 = np.array(nojitnn64)
    nojitnn32 = np.array(nojitnn32)

    rel_jit64 = (jit64 / (Ns*Ns))
    rel_jit32 = (jit32 / (Ns*Ns))
    rel_nojit64 = (nojit64 / (Ns*Ns))
    rel_nojit32 = (nojit32 / (Ns*Ns))
    rel_idw64 = (idw64 / (Ns*Ns))
    rel_idw32 = (idw32 / (Ns*Ns))
    rel_nojitidw64 = (nojitidw64 / (Ns*Ns))
    rel_nojitidw32 = (nojitidw32 / (Ns*Ns))
    rel_nn64 = (nn64 / (Ns*Ns))
    rel_nn32 = (nn32 / (Ns*Ns))
    rel_nojitnn64 = (nojitnn64 / (Ns*Ns))
    rel_nojitnn32 = (nojitnn32 / (Ns*Ns))

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
    ax.loglog(Ns, rel_nn64, '--sg', label='NN-64')
    ax.loglog(Ns, rel_nn32, '--sr', label='NN-32')
    ax.loglog(Ns, rel_nojitnn64, '--sb', label='NOJIT-NN-64')
    ax.loglog(Ns, rel_nojitnn32, '--sy', label='NOJIT-NN-32')
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)
    plt.legend(loc=1)
    plt.xlabel('N')
    plt.ylabel('Time per Element')
    plt.savefig("2d_speed_comparison_rel.png")


def visualize_function():
    # Plot individual function
    N=128
    x, y = np.mgrid[x_min:x_max:1j*N,
                    y_min:y_max:1j*N]

    points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
    points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')

    power = 128 # to exaggerate visualization of IDW
    buff = triangle.sample('nearest', points_f64, jit=True)
    buff.shape = x.shape

    plt.figure()
    plt.imshow(buff.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
               interpolation='nearest')
    plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
    plt.colorbar()
    plt.savefig('output_2d.png')


speed_comparison()
# visualize_function()