import numpy as np
import numpy.linalg as npla
import numba as nb
from node import Node2D
from element import Elem2D
import matplotlib.pyplot as plt


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

N = 128
x, y = np.mgrid[x_min:x_max:1j*N,
                y_min:y_max:1j*N]

points_f64 = np.array([_.ravel() for _ in (x,y)], dtype='f8')
points_f32 = np.array([_.ravel() for _ in (x,y)], dtype='f')


nonjit_buff = triangle.sample('linear', points_f64)
nonjit_buff = np.ma.filled(nonjit_buff, fill_value=-1.)

jit_buff = triangle.sample('linear', points_f64, jit=True)

diff = nonjit_buff - jit_buff
diff.shape = x.shape

print(npla.norm(diff, ord=2))

fig, ax = plt.subplots()
plt.imshow(diff.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
           interpolation='nearest')
plt.colorbar()
plt.show()
