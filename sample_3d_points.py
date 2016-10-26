from node import Node3D
from element import Elem3D
import element
import numpy as np
import matplotlib.pyplot as plt
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
    jit64 = []
    jit32 = []
    nojit32 = []
    nojit64 = []
    idw64 = []
    idw32 = []
    nojitidw64 = []
    nojitidw32 = []
    Ns = np.array([8, 16, 32, 64, 128, 256])

    for N in Ns:
        x, y, z = np.mgrid[x_min:x_max:1j*N,
                           y_min:y_max:1j*N,
                           z_min:z_max:1j*N]

        points_f64 = np.array([_.ravel() for _ in (x, y, z)], dtype="f8")
        points_f32 = np.array([_.ravel() for _ in (x, y, z)], dtype="f")

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
    plt.savefig("3d_speed_comparison.png")


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
    plt.legend(loc=2)
    plt.xlabel('N')
    plt.ylabel('Time per Element')
    plt.savefig("3d_speed_comparison_rel.png")


def test_individual():
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

