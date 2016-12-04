from feainterp.node import Node as Node
from feainterp.element import Element as Element
import numpy as np
import numpy.linalg as npla
import numba as nb


# 2D sample points
p1 = np.array([2, 2], dtype='f8')
p2 = np.array([4, 3], dtype='f8')
p3 = np.array([1, 4], dtype='f8')
p1_f32 = p1.astype('f')
p2_f32 = p2.astype('f')
p3_f32 = p3.astype('f')
v1 = 1.
v2 = 2.
v3 = 3.

trans = np.array([[p2[0]-p1[0], p3[0]-p1[0]],
                  [p2[1]-p1[1], p3[1]-p1[1]]], dtype='f8')
trans = npla.inv(trans)
trans_f32 = trans.astype('f')

node1 = Node(p1, v1)
node2 = Node(p2, v2)
node3 = Node(p3, v3)
triangle = Element([node1, node2, node3])

x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))

N = 2048
x, y = np.mgrid[x_min:x_max:1j*N,
                y_min:y_max:1j*N]
points_f64 = np.array([_.ravel() for _ in (x, y)], dtype='f8').T
points_f32 = np.array([_.ravel() for _ in (x, y)], dtype='f').T
power = 128 # to exaggerate visualization of IDW