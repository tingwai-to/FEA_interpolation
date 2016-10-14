from __future__ import print_function
import numpy as np
import numpy.linalg as npla
from numba import jit
import numba as nb


def make_3d_jit(dtype):
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


def make_3d_nojit(dtype):
    def linear_3d(p1, p2, p3, p4, point, v1, v2, v3, v4):
        """Non-JIT linear interpolation for 3D element"""
        trans = np.array([[p2[0]-p1[0], p3[0]-p1[0], p4[0]-p1[0]],
                          [p2[1]-p1[1], p3[1]-p1[1], p4[1]-p1[1]],
                          [p2[2]-p1[2], p3[2]-p1[2], p4[2]-p1[2]]], dtype=dtype)
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
    return linear_3d


def make_2d_jit(dtype):
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
                v_point[j] = v1 * (area1 / 0.5) + v2 * (area2 / 0.5) + v3 * (area3 / 0.5)

        return v_point
    return linear_2d


def make_2d_nojit(dtype):
    def linear_2d(p1, p2, p3, point, v1, v2, v3):
        """Non-JIT linear interpolation for 2D element"""
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
    return linear_2d


linear_2d_f64 = make_2d_jit(nb.float64)
linear_2d_f32 = make_2d_jit(nb.float32)
linear_2d_nojit64 = make_2d_nojit(np.float64)
linear_2d_nojit32 = make_2d_nojit(np.float32)

linear_3d_f64 = make_3d_jit(nb.float64)
linear_3d_f32 = make_3d_jit(nb.float32)
linear_3d_nojit64 = make_3d_nojit(np.float64)
linear_3d_nojit32 = make_3d_nojit(np.float32)
