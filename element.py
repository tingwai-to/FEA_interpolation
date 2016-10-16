from __future__ import print_function
import numpy as np
import numpy.linalg as npla
from numpy import sqrt
from numba import jit
import numba as nb


class Element(object):
    def __init__(self):
        raise NotImplementedError('Abstract class')


class Elem3D(Element):
    def __init__(self, node1, node2, node3, node4):
        self.p1 = node1.coord
        self.p2 = node2.coord
        self.p3 = node3.coord
        self.p4 = node4.coord
        self.v1 = node1.value
        self.v2 = node2.value
        self.v3 = node3.value
        self.v4 = node4.value


class Elem2D(Element):
    def __init__(self, node1, node2, node3):
        self.p1 = node1.coord
        self.p2 = node2.coord
        self.p3 = node3.coord
        self.v1 = node1.value
        self.v2 = node2.value
        self.v3 = node3.value

    # def sample(self, method, dtype, point, **kwargs):
    #     if method == 'linear':
    #         return self.linear_nojit(dtype, point)
    #     elif method == 'idw':
    #         if 'power' in kwargs:
    #             return self.idw_nojit(dtype, point, kwargs['power'])
    #         else:
    #             raise ValueError('missing IDW argument: power')
    #     else:
    #         raise AttributeError('no interpolation function called ' + method)


    def linear_jit(self, dtype, point):
        # TODO: fix dtype, method not complete
        @jit(dtype[:](dtype[:], dtype[:], dtype[:], dtype[:, :],
                      dtype, dtype, dtype),
             nopython=True)
        def linear(p1, p2, p3, point, v1, v2, v3):
            """JIT optimized linear interpolation for 2D element"""
            # Transformation matrix to reference element
            trans = np.empty((2,2), dtype=dtype)
            trans[0,0] = p2[0] - p1[0]
            trans[0,1] = p3[0] - p1[0]
            trans[1,0] = p2[1] - p1[1]
            trans[1,1] = p3[1] - p1[1]
            trans = npla.inv(trans)

            v_point = np.empty(point.shape[1], dtype=dtype)
            for j in range(point.shape[1]):
                # Transform points to new space
                ref_point_x = trans[0,0]*(point[0,j]-p1[0]) + \
                              trans[0,1]*(point[1,j]-p1[0])
                ref_point_y = trans[1,0]*(point[0,j]-p1[1]) + \
                              trans[1,1]*(point[1,j]-p1[1])
                area2 = 0.5*ref_point_x
                area3 = 0.5*ref_point_y
                area1 = 0.5 - area2 - area3

                if (area1/0.5) < 0 or \
                   (area1/0.5) > 1 or \
                   (area2/0.5) < 0 or \
                   (area2/0.5) > 1 or \
                   (area3/0.5) < 0 or \
                   (area3/0.5) > 1:
                    v_point[j] = -1
                else:
                    v_point[j] = v1*(area1/0.5) + v2*(area2/0.5) + v3*(area3/0.5)
            return v_point
        return linear(self.p1, self.p2, self.p3, point, self.v1, self.v2, self.v3)

    def linear_nojit(self, dtype, point):
        """Non-JIT linear interpolation for 2D self"""
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3

        trans = np.array([[p2[0]-p1[0], p3[0]-p1[0]],
                          [p2[1]-p1[1], p3[1]-p1[1]]], dtype=dtype)
        trans = npla.inv(trans)

        ref_point = trans.dot(np.array([point[0]-p1[0], point[1]-p1[1]], dtype=dtype))

        tot_area = np.array([0.5], dtype=dtype)
        area2 = np.array([0.5*1], dtype=dtype)*ref_point[0]
        area3 = np.array([0.5*1], dtype=dtype)*ref_point[1]
        area1 = tot_area - area2 - area3

        v_point = v1*(area1/tot_area) + v2*(area2/tot_area) + v3*(area3/tot_area)

        # mask = np.ones_like(v_point, dtype="bool")
        # for a in [area1, area2, area3]:
        #     mask *= (a/tot_area) > 0
        #     mask *= (a/tot_area) < 1

        return v_point#, mask

    def idw_nojit(self, dtype, point, power):
        """Non-JIT simple inverse distance weighting"""
        def distance(xn, x):
            return sqrt((xn[0] - x[0]) ** 2 + (xn[1] - x[1]) ** 2)

        def weight(xn, x, p):
            return 1 / distance(xn, x) ** p

        p1 = self.p1.astype(dtype)
        p2 = self.p2.astype(dtype)
        p3 = self.p3.astype(dtype)
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3

        v_point = (v1 * weight(p1, point, power) +
                   v2 * weight(p2, point, power) +
                   v3 * weight(p3, point, power)) / \
                  (weight(p1, point, power) +
                   weight(p2, point, power) +
                   weight(p3, point, power))
        return v_point
