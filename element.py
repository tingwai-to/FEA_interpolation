from __future__ import print_function
import numpy as np
import numpy.linalg as npla
from numpy import sqrt
from numba import jit
import numba as nb


class Element(object):
    def __init__(self):
        raise NotImplementedError('Abstract class')

    def sample(self, method, point, *args, **kwargs):
        use_jit = kwargs.pop("jit", False)
        if use_jit:
            funcs = jit_functions['%sd' % self.dimensionality][method]
            func = funcs[nb.numpy_support.from_dtype(point.dtype)]
            return self._call_jit(func, point, *args, **kwargs)
        else:
            func = getattr(self, '%s_nojit' % method, None)
            if func is None:
                raise AttributeError('no interpolation function called %s' % method)
            return func(point, *args, **kwargs)


class Elem3D(Element):
    dimensionality = 3
    def __init__(self, node1, node2, node3, node4):
        self.p1 = node1.coord
        self.p2 = node2.coord
        self.p3 = node3.coord
        self.p4 = node4.coord
        self.v1 = node1.value
        self.v2 = node2.value
        self.v3 = node3.value
        self.v4 = node4.value

    def linear_nojit(self, point):
        """Non-JIT linear interpolation for 3D element"""
        dtype = point.dtype
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        p4 = self.p4
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3
        v4 = self.v4

        trans = np.array([[p2[0]-p1[0], p3[0]-p1[0], p4[0]-p1[0]],
                          [p2[1]-p1[1], p3[1]-p1[1], p4[1]-p1[1]],
                          [p2[2]-p1[2], p3[2]-p1[2], p4[2]-p1[2]]],
                         dtype=dtype)
        trans = npla.inv(trans)

        ref_point = trans.dot(np.array([point[0]-p1[0],
                                        point[1]-p1[1],
                                        point[2]-p1[2]],
                                       dtype=dtype))

        tot_vol = np.array([1./6], dtype=dtype)  # Volume of tetrahedron
        vol2 = np.array([(1./3)*(1./2)], dtype=dtype)*ref_point[0]
        vol3 = np.array([(1./3)*(1./2)], dtype=dtype)*ref_point[1]
        vol4 = np.array([(1./3)*(1./2)], dtype=dtype)*ref_point[2]
        vol1 = tot_vol - vol2 - vol3 - vol4

        v_point = v1*(vol1/tot_vol) + v2*(vol2/tot_vol) + \
                  v3*(vol3/tot_vol) + v4*(vol4/tot_vol)

        # mask = np.ones_like(v_point, dtype="bool")
        # for v in [vol1, vol2, vol3, vol4]:
        #     mask &= (v/tot_vol) > 0
        #     mask &= (v/tot_vol) < 1

        return v_point#, mask

    def idw_nojit(self, point, power=2):
        """Non-JIT simple inverse distance weighting"""
        def weight(xn, x, p):
            return 1 / self._distance(xn,x)**p

        dtype = point.dtype
        p1 = self.p1.astype(dtype)
        p2 = self.p2.astype(dtype)
        p3 = self.p3.astype(dtype)
        p4 = self.p4.astype(dtype)
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3
        v4 = self.v4

        v_point = (v1*weight(p1, point, power) +
                   v2*weight(p2, point, power) +
                   v3*weight(p3, point, power) +
                   v4*weight(p4, point, power)) / \
                  (weight(p1, point, power) +
                   weight(p2, point, power) +
                   weight(p3, point, power) +
                   weight(p4, point, power))
        return v_point

    def nearest_nojit(self, point):
        """Non-JIT nearest neighbor"""
        dtype = point.dtype
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        p4 = self.p4
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3
        v4 = self.v4
        values = (v1, v2, v3, v4)

        dist = np.empty((4, point.shape[1]), dtype=dtype)
        dist[0] = self._distance(p1, point)
        dist[1] = self._distance(p2, point)
        dist[2] = self._distance(p3, point)
        dist[3] = self._distance(p4, point)

        nearest = np.empty(point.shape[1], dtype=dtype)
        for j in range(dist.shape[1]):
            nearest[j] = values[np.argmin(dist[:,j])]

        return nearest

    def _call_jit(self, func, point, *args, **kwargs):
        dtype = point.dtype
        return func(self.p1.astype(dtype),
                    self.p2.astype(dtype),
                    self.p3.astype(dtype),
                    self.p4.astype(dtype),
                    point,
                    self.v1, self.v2, self.v3, self.v4, *args, **kwargs)

    @staticmethod
    def _distance(xn, x):
        return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2 + (xn[2]-x[2])**2)


class Elem2D(Element):
    dimensionality = 2
    def __init__(self, node1, node2, node3):
        self.p1 = node1.coord
        self.p2 = node2.coord
        self.p3 = node3.coord
        self.v1 = node1.value
        self.v2 = node2.value
        self.v3 = node3.value

    def linear_nojit(self, point):
        """Non-JIT linear interpolation for 2D self"""
        dtype = point.dtype
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3

        trans = np.array([[p2[0]-p1[0], p3[0]-p1[0]],
                          [p2[1]-p1[1], p3[1]-p1[1]]], dtype=dtype)
        trans = npla.inv(trans)

        ref_point = trans.dot(np.array([point[0]-p1[0],
                                        point[1]-p1[1]], dtype=dtype))

        tot_area = np.array([0.5], dtype=dtype)
        area2 = np.array([0.5*1], dtype=dtype)*ref_point[0]
        area3 = np.array([0.5*1], dtype=dtype)*ref_point[1]
        area1 = tot_area - area2 - area3

        v_point = v1*(area1/tot_area) + v2*(area2/tot_area) + v3*(area3/tot_area)

        mask = np.ones_like(v_point, dtype="bool")
        for a in [area1, area2, area3]:
            mask *= (a/tot_area) > 0
            mask *= (a/tot_area) < 1

        v_point = np.ma.MaskedArray(v_point, ~mask)

        return v_point#, mask

    def idw_nojit(self, point, power=2):
        """Non-JIT simple inverse distance weighting"""
        def weight(xn, x, p):
            return 1 / self._distance(xn, x) ** p

        dtype = point.dtype
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

    def nearest_nojit(self, point):
        """Non-JIT nearest neighbor"""
        dtype = point.dtype
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3
        values = (v1, v2, v3)

        dist = np.empty((3, point.shape[1]), dtype=dtype)
        dist[0] = self._distance(p1, point)
        dist[1] = self._distance(p2, point)
        dist[2] = self._distance(p3, point)

        nearest = np.empty(point.shape[1], dtype=dtype)
        for j in range(dist.shape[1]):
            nearest[j] = values[np.argmin(dist[:,j])]

        return nearest

    def _call_jit(self, func, point, *args, **kwargs):
            dtype = point.dtype
            return func(self.p1.astype(dtype),
                        self.p2.astype(dtype),
                        self.p3.astype(dtype),
                        point,
                        self.v1, self.v2, self.v3, *args, **kwargs)

    @staticmethod
    def _distance(xn, x):
        return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2)


def make_3d_lin_jit(dtype):
    @jit(dtype[:](dtype[:], dtype[:], dtype[:], dtype[:], dtype[:,:],
         dtype, dtype, dtype, dtype),
         nopython=True)
    def linear_3d(p1, p2, p3, p4, point, v1, v2, v3, v4):
        """JIT optimized linear interpolation for 3D element"""
        # Transformation matrix to reference element
        trans = np.empty((3,3), dtype=dtype)
        trans[0,0] = p2[0]-p1[0]
        trans[0,1] = p3[0]-p1[0]
        trans[0,2] = p4[0]-p1[0]
        trans[1,0] = p2[1]-p1[1]
        trans[1,1] = p3[1]-p1[1]
        trans[1,2] = p4[1]-p1[1]
        trans[2,0] = p2[2]-p1[2]
        trans[2,1] = p3[2]-p1[2]
        trans[2,2] = p4[2]-p1[2]
        trans = npla.inv(trans)

        v_point = np.empty(point.shape[1], dtype=dtype)
        for j in range(point.shape[1]):
            # Transform points to new space
            ref_point_x = trans[0,0]*(point[0,j]-p1[0]) + \
                          trans[0,1]*(point[1,j]-p1[0]) + \
                          trans[0,2]*(point[2,j]-p1[0])
            ref_point_y = trans[1,0]*(point[0,j]-p1[1]) + \
                          trans[1,1]*(point[1,j]-p1[1]) + \
                          trans[1,2]*(point[2,j]-p1[1])
            ref_point_z = trans[2,0]*(point[0,j]-p1[2]) + \
                          trans[2,1]*(point[1,j]-p1[2]) + \
                          trans[2,2]*(point[2,j]-p1[2])
            vol2 = 1./6*ref_point_x
            vol3 = 1./6*ref_point_y
            vol4 = 1./6*ref_point_z
            vol1 = 1./6 - vol2 - vol3 - vol4

            if vol1/(1./6) < 0 or \
               vol1/(1./6) > 1 or \
               vol2/(1./6) > 0 or \
               vol2/(1./6) > 1 or \
               vol3/(1./6) > 0 or \
               vol3/(1./6) > 1:
                v_point[j] = -1
            else:
                v_point[j] = v1*(vol1/(1./6)) + v2*(vol2/(1./6)) + \
                             v3*(vol3/(1./6)) + v4*(vol4/(1./6))

        # mask = np.ones_like(v_point, dtype=nb.uint8)
        # for v in [vol1, vol2, vol3, vol4]:
        #     mask &= (v/tot_vol) > 0
        #     mask &= (v/tot_vol) < 1

        return v_point#, mask
    return linear_3d


def make_2d_lin_jit(dtype):
    @jit(dtype[:](dtype[:], dtype[:], dtype[:], dtype[:,:],
         dtype, dtype, dtype),
         nopython=True)
    def linear_2d(p1, p2, p3, point, v1, v2, v3):
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
    return linear_2d


def make_3d_idw_jit(dtype):
    @jit(dtype[:](dtype[:], dtype[:,:]), nopython=True)
    def distance(xn, x):
        """Distance of nodes to points"""
        return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2 + (xn[2]-x[2])**2)

    @jit(dtype[:](dtype[:], dtype[:,:], nb.int32), nopython=True)
    def weight(xn, x, p):
        """Weight of nodes to point"""
        return 1 / distance(xn, x)**p

    @jit(dtype[:](dtype[:], dtype[:], dtype[:], dtype[:], dtype[:,:],
         dtype, dtype, dtype, dtype, nb.int32),
         nopython=True)
    def simple_3d(p1, p2, p3, p4, point, v1, v2, v3, v4, power):
        """Simple inverse distance weighting function"""
        v_point = (v1*weight(p1, point, power) +
                   v2*weight(p2, point, power) +
                   v3*weight(p3, point, power) +
                   v4*weight(p4, point, power)) / \
                  (weight(p1, point, power) +
                   weight(p2, point, power) +
                   weight(p3, point, power) +
                   weight(p4, point, power))
        return v_point
    return simple_3d


def make_2d_idw_jit(dtype):
    @jit(dtype[:](dtype[:], dtype[:,:]), nopython=True)
    def distance(xn, x):
        """Distance of nodes to points"""
        return sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2)

    @jit(dtype[:](dtype[:], dtype[:,:], nb.int32), nopython=True)
    def weight(xn, x, p):
        """Weight of nodes to point"""
        return 1 / distance(xn, x)**p

    @jit(dtype[:](dtype[:], dtype[:], dtype[:], dtype[:,:],
         dtype, dtype, dtype, nb.int32),
         nopython=True)
    def simple_2d(p1, p2, p3, point, v1, v2, v3, power):
        """Simple inverse distance weighting function"""
        v_point = (v1*weight(p1, point, power) +
                   v2*weight(p2, point, power) +
                   v3*weight(p3, point, power)) / \
                  (weight(p1, point, power) +
                   weight(p2, point, power) +
                   weight(p3, point, power))
        return v_point
    return simple_2d

jit_functions = {}

jit_functions['3d'] = {}
jit_functions['3d']['linear'] = {}
jit_functions['3d']['linear'][nb.float64] = make_3d_lin_jit(nb.float64)
jit_functions['3d']['linear'][nb.float32] = make_3d_lin_jit(nb.float32)
jit_functions['3d']['idw'] = {}
jit_functions['3d']['idw'][nb.float64] = make_3d_idw_jit(nb.float64)
jit_functions['3d']['idw'][nb.float32] = make_3d_idw_jit(nb.float32)

jit_functions['2d']= {}
jit_functions['2d']['linear'] = {}
jit_functions['2d']['linear'][nb.float64] = make_2d_lin_jit(nb.float64)
jit_functions['2d']['linear'][nb.float32] = make_2d_lin_jit(nb.float32)
jit_functions['2d']['idw'] = {}
jit_functions['2d']['idw'][nb.float64] = make_2d_idw_jit(nb.float64)
jit_functions['2d']['idw'][nb.float32] = make_2d_idw_jit(nb.float32)
