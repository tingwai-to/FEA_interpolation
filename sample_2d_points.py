import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
from test_data import *
import os
import sys


def speed_comparison():
    # JIT vs non-JIT
    fig, ax = plt.subplots()
    funcs = [('linear', 64, True), #('linear', 32, True),
             ('linear', 64, False), #('linear', 32, False),
             ('idw', 64, True), #('idw', 32, True),
             ('idw', 64, False), #('idw', 32, False),
             ('nearest', 64, True), #('nearest', 32, True),
             ('nearest', 64, False), #('nearest', 32, False)
             ]

    Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    times = np.empty((len(funcs), Ns.shape[0]))

    for i, N in enumerate(Ns):
        print('N = %i' % N)
        x, y = np.mgrid[x_min:x_max:1j*N,
                        y_min:y_max:1j*N]
        points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8').T
        points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f').T

        for j, fun in enumerate(funcs):
            print(fun)
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
    result = triangle.sample('linear', points_f64, jit='cpu')
    result_nojit = triangle.sample('linear', points_f64, jit=False)
    result.shape = x.shape
    result_nojit.shape = x.shape
    diff = result.T - result_nojit.T

    print(np.sum(diff))
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
        buff = triangle.sample('nearest', points_f64, jit=True)
    end = time()
    print('JIT:  %i times took %.5f seconds' % (N, end-start))

    start = time()
    for i in range(N):
        buff = triangle.sample('nearest', points_f64, jit=False)
    end = time()
    print('No-JIT: %i times took %.5f seconds' % (N, end-start))


# speed_comparison()
visualize_function()
# time_function()
