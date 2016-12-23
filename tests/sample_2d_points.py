import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
from example_data import *
import os
import sys

from feainterp.element import jit_functions
from numba import cuda

def speed_comparison():
    # JIT vs non-JIT
    fig, ax = plt.subplots()
    funcs = [('linear', 64, 'cpu'),
             ('linear', 64, 'gpu'),
             ('linear', 64, False),
             # ('idw', 64, 'cpu'),
             # ('idw', 64, 'gpu'),
             # ('idw', 64, False),
             # ('nearest', 64, 'cpu'),
             # ('nearest', 64, 'gpu'),
             # ('nearest', 64, False),
             ]

    Ns = np.array([64, 128, 256, 512, 1024, 2048, 4096])
    times = np.zeros((len(funcs), Ns.shape[0]))
    repeat = 20

    for i, N in enumerate(Ns):
        print('N = %i' % N)
        x, y = np.mgrid[x_min:x_max:1j*N,
                        y_min:y_max:1j*N]
        points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8').T
        points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f').T

        for j, fun in enumerate(funcs):
            print(fun)
            result = np.empty((points_f64.shape[0]), dtype=points_f64.dtype)

            # dirty hack to time functions
            if fun[2] == 'cpu':
                start = time()
                func = jit_functions['2d']['cpu_'+fun[0]]
                for again in range(repeat):
                    func(result, p1, p2, p3, points_f64, v1, v2, v3,
                         trans  # linear
                         # 2  # idw
                         )
                end = time()
                print (end-start)
                times[j, i] += (end-start)/repeat

            if fun[2] == 'gpu':
                gpu_result = cuda.to_device(result)
                gp1 = cuda.to_device(p1)
                gp2 = cuda.to_device(p2)
                gp3 = cuda.to_device(p3)
                gpoints = cuda.to_device(points_f64)
                gtrans = cuda.to_device(trans)
                threadsperblock = 32
                blockspergrid = (gpu_result.size + (threadsperblock - 1)) // threadsperblock
                start = time()
                func = jit_functions['2d']['gpu_'+fun[0]]
                for again in range(repeat):
                    func[blockspergrid, threadsperblock]\
                    (gpu_result, gp1, gp2, gp3, gpoints, v1, v2, v3,
                     gtrans  # linear
                     # 2  #idw
                     )
                end = time()
                result = gpu_result.copy_to_host()
                print (end-start)
                times[j, i] += (end-start)/repeat

            if fun[2] == False:
                start = time()
                for again in range(repeat):
                    triangle.linear_nojit(points_f64)
                end = time()
                print (end-start)
                times[j, i] += (end-start)/repeat

    # colors = cm.nipy_spectral(np.linspace(0, 1, len(funcs)))
    for c, row in enumerate(times):
        ax.loglog(Ns, row,
                  # color=colors[c],
                  marker='o', ls='-',
                  label='%s, float%d, jit=%r' % (funcs[c][0], funcs[c][1], funcs[c][2]))
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)
    plt.legend(loc=2, prop={'size':16}, frameon=False)
    plt.xlabel('N$^2$ samples', fontsize=16)
    plt.ylabel('Total time [sec]', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    xtop = ax.get_xlim()
    ytop = ax.get_ylim()
    ax.set_xlim(xtop[0]*.95, xtop[1]*1.05)
    ax.set_ylim(top=ytop[1]*1.1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.15)
    plt.savefig("2d_speed_comparison.png")

    rel_times = times / (Ns * Ns)

    plt.clf()
    ax = plt.gca()
    for c, row in enumerate(rel_times):
        ax.loglog(Ns, row,
                  # color=colors[c],
                  marker='o', ls='-',
                  label='%s, float%d, jit=%r' % (funcs[c][0], funcs[c][1], funcs[c][2]))
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=10)
    plt.legend(loc=3, prop={'size':16}, frameon=False)
    plt.xlabel('N$^2$ samples', fontsize=16)
    plt.ylabel('Time per sample [sec]', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    xtop = ax.get_xlim()
    ax.set_xlim(xtop[0]*.95, xtop[1]*1.05)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.15)
    plt.savefig("2d_speed_comparison_rel.png")

    # repeat for all shape functions
    speedup = times[-1] / times[:-1]  # divides cpu/gpu by non-jit time
    import csv
    with open("speedup.csv", 'a+b') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(speedup)


def speedup_plot():
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height >= 10:
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        '%d' % int(height),
                        ha='center', va='bottom', fontsize=12)
            else:
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        '%0.1f' % float(height),
                        ha='center', va='bottom', fontsize=12)

    # run this function after benchmarking with speed_comparison()
    # very messy but makes nice looking plots
    from numpy import genfromtxt
    speedup = genfromtxt('speedup.csv', delimiter=',')
    ind = np.arange(speedup.shape[1])
    width = 0.28

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, speedup[0], width, color='#e41a1c')  # linear
    rects3 = ax.bar(ind + width, speedup[4], width, color='#377eb8')  # nearest
    rects2 = ax.bar(ind + width*2, speedup[2], width, color='#4daf4a')  # idw

    ax.set_ylabel('Speedup')
    ax.set_xticks(ind + width*1.5)
    ax.set_xticklabels([64, 128, 256, 512, 1024, 2048, 4096])
    ax.set_xlabel('N$^2$ samples')
    ax.legend((rects1[0], rects3[0], rects2[0]), ('Linear', 'Nearest', 'IDW'), loc=2,
              prop={'size':12}, frameon=False)
    ytop = ax.get_ylim()[1]
    ax.set_ylim(top=ytop*1.05)
    plt.title('CPU JIT Speedup', size=18)

    autolabel(rects1)
    autolabel(rects3)
    autolabel(rects2)
    fig.set_size_inches(12,6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    fig.set_size_inches(12,6)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.1)
    plt.savefig("speed_cpu.png")


    plt.clf()
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, speedup[1], width, color='#e41a1c')  # linear
    rects3 = ax.bar(ind + width, speedup[5], width, color='#377eb8')  # nearest
    rects2 = ax.bar(ind + width*2, speedup[3], width, color='#4daf4a')  # idw

    ax.set_ylabel('Speedup')
    ax.set_xticks(ind + width*1.5)
    ax.set_xticklabels([64, 128, 256, 512, 1024, 2048, 4096])
    ax.set_xlabel('N$^2$ samples')
    ax.legend((rects1[0], rects3[0], rects2[0]), ('Linear', 'Nearest', 'IDW'), loc=2,
              prop={'size':12}, frameon=False)
    ytop = ax.get_ylim()[1]
    ax.set_yscale('log', basey=10)
    ax.set_ylim(top=ytop*2)
    plt.title('GPU JIT Speedup', size=18)

    autolabel(rects1)
    autolabel(rects3)
    autolabel(rects2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    fig.set_size_inches(12,6)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.1)
    plt.savefig("speed_gpu.png")


def visualize_function():
    # Plot individual function
    result = triangle.sample('nearest', points_f64, jit='gpu')
    result_nojit = triangle.sample('nearest', points_f64, jit=False)
    result.shape = x.shape
    result_nojit.shape = x.shape
    diff = result.T - result_nojit.T

    print(np.sum(diff))
    plt.figure()
    plt.imshow(diff, extent=[x_min, x_max, y_min, y_max], origin='lower',
               interpolation='nearest')
    plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    # plt.show()
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


speed_comparison()
# speedup_plot()
# visualize_function()
# time_function()
