from node import Node3D
from element import Elem3D
import element
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numba as nb
from time import time
import os
import sys


# 3D sample points
p1 = np.array([2,2,1], dtype="f8")
p2 = np.array([4,3.5,5], dtype="f8")
p3 = np.array([1,4,3], dtype="f8")
p4 = np.array([6,5,3], dtype="f8")
v1 = 0.7
v2 = 2.2
v3 = 3.4
v4 = 4.6

node1 = Node3D(p1, v1)
node2 = Node3D(p2, v2)
node3 = Node3D(p3, v3)
node4 = Node3D(p4, v4)
triangle = Elem3D(node1, node2, node3, node4)

x_min = min(_[0] for _ in (p1, p2, p3, p4))
y_min = min(_[1] for _ in (p1, p2, p3, p4))
z_min = min(_[2] for _ in (p1, p2, p3, p4))
x_max = max(_[0] for _ in (p1, p2, p3, p4))
y_max = max(_[1] for _ in (p1, p2, p3, p4))
z_max = max(_[2] for _ in (p1, p2, p3, p4))


def speed_comparison():
    # JIT vs non-JIT
    fig, ax = plt.subplots()
    funcs = [('linear', 64, True), ('linear', 32, True),
             ('linear', 64, False), ('linear', 32, False),
             ('idw', 64, True), ('idw', 32, True),
             ('idw', 64, False), ('idw', 32, False),
             # ('nearest', 64, True), ('nearest', 32, True),
             ('nearest', 64, False), ('nearest', 32, False)]

    Ns = np.array([8, 16, 32, 64, 128, 256])
    times = np.empty((len(funcs), Ns.shape[0]))

    for i, N in enumerate(Ns):
        x, y, z = np.mgrid[x_min:x_max:1j * N,
                           y_min:y_max:1j * N,
                           z_min:z_max:1j * N]
        points_f64 = np.array([_.ravel() for _ in (x, y, z)], dtype="f8")
        points_f32 = np.array([_.ravel() for _ in (x, y, z)], dtype="f")

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
    plt.savefig("3d_speed_comparison.png")

    rel_times = times / (Ns * Ns)

    plt.clf()
    ax = plt.gca()
    for c, row in enumerate(rel_times):
        ax.loglog(Ns, row, color=colors[c], marker='o', ls='-',
                  label='%s-%d, jit=%r' % (funcs[c][0], funcs[c][1], funcs[c][2]))
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)
    plt.legend(loc=1, prop={'size':10})
    plt.xlabel('N')
    plt.ylabel('Time per Element')
    plt.savefig("3d_speed_comparison_rel.png")


def visualize_function():
    # Plot individual function
    N=128
    x, y, z = np.mgrid[x_min:x_max:1j*N,
                       y_min:y_max:1j*N,
                       z_min:z_max:1j*N]

    points_f64 = np.array([_.ravel() for _ in (x, y, z)], dtype="f8")
    points_f32 = np.array([_.ravel() for _ in (x, y, z)], dtype="f")

    buff = triangle.sample('nearest', points_f64)
    buff.shape = x.shape
    # mask.shape = x.shape
    # buffer = np.ma.MaskedArray(buff, ~mask).transpose()

    if not os.path.isdir("frames3d"):
        os.mkdir("frames3d")

    for i in range(N):
        print(i)

        plt.clf()
        plt.imshow(buff[i,:,:], origin='lower', interpolation='nearest',
                   extent=[y_min, y_max, z_min, z_max])
        plt.clim(1, 4)
        plt.colorbar()
        plt.savefig("frames/slice_x_%03i.png" % i)

        plt.clf()
        plt.imshow(buff[:,i,:], origin='lower', interpolation='nearest',
                   extent=[x_min, x_max, z_min, z_max])
        plt.clim(1, 4)
        plt.colorbar()
        plt.savefig("frames/slice_y_%03i.png" % i)

        plt.clf()
        plt.imshow(buff[:,:,i], origin='lower', interpolation='nearest',
                   extent=[x_min, x_max, y_min, y_max])
        plt.clim(1, 4)
        plt.colorbar()
        plt.savefig("frames/slice_z_%03i.png" % i)


speed_comparison()
# visualize_function()
