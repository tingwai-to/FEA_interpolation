from __future__ import print_function
import numpy as np
import numpy.linalg as npla
from numpy import sqrt


class Element2D(object):
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
