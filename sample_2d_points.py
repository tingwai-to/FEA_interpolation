from node import Node
from element import Element
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

node1 = Node(p1, v1)
node2 = Node(p2, v2)
node3 = Node(p3, v3)
triangle = Element([node1, node2, node3])

node1_2d = Node2D(p1, v1)
node2_2d = Node2D(p2, v2)
node3_2d = Node2D(p3, v3)
triangle2d = Elem2D(node1_2d, node2_2d, node3_2d)


x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))

N=128
x, y = np.mgrid[x_min:x_max:1j*N,
                y_min:y_max:1j*N]
points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8')
points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f')
power = 128 # to exaggerate visualization of IDW


def test_cuda():
    p1 = triangle.p1
    p2 = triangle.p2
    p3 = triangle.p3
    v1 = triangle.v1
    v2 = triangle.v2
    v3 = triangle.v3
    N = 1024
    x, y = np.mgrid[x_min:x_max:1j*N,
                    y_min:y_max:1j*N]

    trans = np.empty((2,2), dtype='f8')
    trans[0,0] = p2[0] - p1[0]
    trans[0,1] = p3[0] - p1[0]
    trans[1,0] = p2[1] - p1[1]
    trans[1,1] = p3[1] - p1[1]
    trans = np.linalg.inv(trans)

    point = np.array([_.ravel() for _ in (x, y)], dtype='f8')
    point = point.reshape((N,N,2))
    result = np.empty((N,N), dtype='f8')

    start = time()
    element.make_cuda[1,N](result, p1, p2, p3, point, v1, v2, v3, trans)
    end = time()
    print(end-start)
    print(result)

    plt.figure()
    plt.imshow(result.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
               interpolation='nearest')
    plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
    plt.colorbar()
    plt.show()


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
        print('N = %i' % N)
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
    plt.savefig("2d_speed_comparison_rel.png")


def visualize_function():
    # Plot individual function

    buff = triangle.sample('nearest', points_f32, jit=False)
    buff_2d = triangle2d.sample('nearest', points_f32, jit=False)
    buff.shape = x.shape
    buff_2d.shape = x.shape
    diff = buff.T - buff_2d.T

    plt.figure()
    plt.imshow(diff, extent=[x_min, x_max, y_min, y_max], origin='lower',
               interpolation='nearest')
    plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
    plt.colorbar()
    plt.savefig('output_2d.png')


def time_function():
    N=100
    start = time()
    for i in range(N):
        # Elem2D class
        buff_2d = triangle2d.sample('nearest', points_f32, jit=False)
    end = time()
    print('Elem2D:  %i times took %.5f seconds' % (N, end-start))

    start = time()
    for i in range(N):
        # Element class
        buff = triangle.sample('nearest', points_f32, jit=False)
    end = time()
    print('Element: %i times took %.5f seconds' % (N, end-start))


# test_cuda()
# speed_comparison()
visualize_function()
time_function()