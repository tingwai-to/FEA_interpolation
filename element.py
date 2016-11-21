from __future__ import print_function
import numpy as np
import numpy.linalg as npla
from numpy import sqrt
from numba import jit
from numba import cuda
import numba as nb
from math import factorial
from scipy.spatial.distance import cdist
import jit_setup


class Element(object):
    def __init__(self, nodes):
        """Create an element.

        Args:
            nodes (list of Node): Dimension of nodes in list must be the same.
                Number of nodes in list (n) should equal desired dimension of
                Element plus one. ie. n = dim + 1
        """
        #: int: dimensionality of element, derived from Node
        self.dim = nodes[0].dim
        #: list of np.array: coordinates of nodes
        self.coords = np.array([thisnode.coord for thisnode in nodes])
        #: list of float: values of nodes
        self.values = np.array([thisnode.value for thisnode in nodes])

        trans = (self.coords[1:,:] - self.coords[0,:]).T
        #: np.array: transformation matrix for linear interpolation
        self.trans = npla.inv(trans)

    def sample(self, method, point, *args, **kwargs):
        """Calls interpolation function

        Args:
            method (str): Desired interpolation style (linear, idw, nearest)
            point (np.array): Array of n-number of points to be interpolated
                eg. where dim=2:
                point = np.array([[x0, y0],
                                  [x1, y1],
                                  [.., ..].
                                  [xn, yn]])

            **kwargs:
                jit (str, bool): Desired compilation method (cpu, gpu)
                    Defaults to False
                power (int): power parameter of for inverse distance weighting
                    Positive integer. Defaults to 2 if not provided

        Returns:
            func: non-JIT interpolation function called if jit argument not provided
                This will then return a np.array of interpolated values
            _call_cuda: called if jit='gpu'
            _call_jit: called if jit='cpu'

        Raises:
            AttributeError: desired interpolation style not supported
            ValueError: interpolation in Element.dim not supported
        """
        target = kwargs.pop("jit", False)
        if target:
            func = jit_functions.get('%sd' % self.dim, None)
            if func is None:
                raise ValueError('JIT in %sd not supported' % self.dim)
            func = func.get(target+'_'+method, None)
            if func is None:
                raise AttributeError('%s %s function not supported' % (target, method))

            if method == 'idw':
                kwargs.setdefault('power', 2)
            elif method == 'linear':
                kwargs['trans'] = self.trans

            result = np.empty((point.shape[0]), dtype=point.dtype)

            if target == 'gpu':
                return self._call_cuda(func, result, point, method, *args, **kwargs)
            return self._call_jit(func, result, point, *args, **kwargs)
        else:
            func = getattr(self, '%s_nojit' % method, None)
            if func is None:
                raise AttributeError('%s function not supported' % method)
            return func(point, *args, **kwargs)

    def _call_jit(self, func, result, point, *args, **kwargs):
        """Calls appropriate CPU JIT kernel.

        Args:
            func (function): CPU JIT kernel
            result (np.array): 1-D array of same length as point, values will
                be populated by func
            point (np.array): Array of n-number of points to be interpolated

            **kwargs:
                power (int): power parameter of for inverse distance weighting
                trans (np.array): transformation matrix, only provide if
                    interpolation style is linear

        Returns:
            result: array of interpolated values at sampled points
        """
        if self.dim == 2:
            func(result,
                 self.coords[0], self.coords[1], self.coords[2],
                 point,
                 self.values[0], self.values[1], self.values[2],
                 *args, **kwargs)
        elif self.dim == 3:
            func(result,
                 self.coords[0], self.coords[1], self.coords[2], self.coords[3],
                 point,
                 self.values[0], self.values[1], self.values[2], self.values[3],
                 *args, **kwargs)
        return result

    def _call_cuda(self, func, result, point, method, *args, **kwargs):
        """Calls appropriate GPU JIT kernel.

        Note:
            CUDA kernels can't accept *args, **kwargs. Arguments like power
            or trans have to be manually passed in.

        Args:
            func (function): CUDA kernel
            result (np.array): 1-D array of same length as point, values will
                be populated by func
            point (np.array): Array of n-number of points to be interpolated

            **kwargs:
                power (int): power parameter of for inverse distance weighting
                trans (np.array): transformation matrix, only provide if
                    interpolation style is linear

        Returns:
            result: array of interpolated values at sampled points
        """
        gpu_result = cuda.to_device(result)
        threadsperblock = 32
        blockspergrid = (gpu_result.size + (threadsperblock - 1)) // threadsperblock
        if self.dim == 2:
            if method == 'linear':
                func[blockspergrid, threadsperblock]\
                    (gpu_result,
                     cuda.to_device(self.coords[0]), cuda.to_device(self.coords[1]), cuda.to_device(self.coords[2]),
                     cuda.to_device(point),
                     self.values[0], self.values[1], self.values[2],
                     cuda.to_device(kwargs['trans']))
            elif method == 'idw':
                func[blockspergrid, threadsperblock]\
                    (gpu_result,
                     cuda.to_device(self.coords[0]), cuda.to_device(self.coords[1]), cuda.to_device(self.coords[2]),
                     cuda.to_device(point),
                     self.values[0], self.values[1], self.values[2],
                     kwargs['power'])
            else:
                func[blockspergrid, threadsperblock]\
                    (gpu_result,
                     cuda.to_device(self.coords[0]), cuda.to_device(self.coords[1]), cuda.to_device(self.coords[2]),
                     cuda.to_device(point),
                     self.values[0], self.values[1], self.values[2])
        result = gpu_result.copy_to_host()
        return result

    def linear_nojit(self, point):
        """Non-JIT linear interpolation for nD element"""
        dtype = point.dtype
        ref_point = self.trans.dot((point-self.coords[0]).T).T

        tot_vol = np.array([1./factorial(self.dim)], dtype=dtype)
        vols = []
        for j in range(self.dim):
            vols.append(tot_vol*ref_point[:,j])
        vols.insert(0, tot_vol - sum(vols))

        v_point = self.values*np.array(vols).T/tot_vol
        v_point = np.sum(v_point, axis=1)

        mask = np.ones_like(v_point, dtype="bool")
        for v in vols:
            mask &= (v/tot_vol) >= 0
            mask &= (v/tot_vol) <= 1

        v_point = np.ma.MaskedArray(v_point, ~mask)
        v_point = v_point.filled(fill_value=-1.)

        return v_point

    def idw_nojit(self, point, power=2):
        """Non-JIT simple inverse distance weighting for nD element"""
        def weight(point, p):
            return 1 / self._distance(point)**p

        weighted_distances = weight(point, power)
        numerator = np.sum(self.values[:,None]*weighted_distances, axis=0)
        denominator =  np.sum(weighted_distances, axis=0)
        v_point = numerator/denominator

        return v_point

    def nearest_nojit(self, point):
        """Non-JIT nearest neighbor for nD element"""
        dist = self._distance(point)

        nearest_index = np.argmin(dist, axis=0)
        nearest = self.values[nearest_index]

        return nearest

    def _distance(self, point):
        """Returns distance between each pair of coords and points"""
        return cdist(self.coords, point)


@jit([nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
                    nb.float32[:,:],
                    nb.float32, nb.float32, nb.float32, nb.float32,
                    nb.float32[:,:]),
      nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    nb.float64[:,:],
                    nb.float64, nb.float64, nb.float64, nb.float64,
                    nb.float64[:,:])],
     nopython=True)
def linear_3d(p1, p2, p3, p4, point, v1, v2, v3, v4, trans):
    """JIT optimized linear interpolation for 3D element"""
    v_point = np.empty(point.shape[0], dtype=point.dtype)
    for i in range(point.shape[0]):
        # Transform points to new space
        ref_point_x = trans[0,0]*(point[i,0]-p1[0]) + \
                      trans[0,1]*(point[i,1]-p1[0]) + \
                      trans[0,2]*(point[i,2]-p1[0])
        ref_point_y = trans[1,0]*(point[i,0]-p1[1]) + \
                      trans[1,1]*(point[i,1]-p1[1]) + \
                      trans[1,2]*(point[i,2]-p1[1])
        ref_point_z = trans[2,0]*(point[i,0]-p1[2]) + \
                      trans[2,1]*(point[i,1]-p1[2]) + \
                      trans[2,2]*(point[i,2]-p1[2])
        vol2 = 1./6*ref_point_x
        vol3 = 1./6*ref_point_y
        vol4 = 1./6*ref_point_z
        vol1 = 1./6 - vol2 - vol3 - vol4

        if vol1/(1./6) < 0 or \
           vol1/(1./6) > 1 or \
           vol2/(1./6) < 0 or \
           vol2/(1./6) > 1 or \
           vol3/(1./6) < 0 or \
           vol3/(1./6) > 1 or \
           vol4/(1./6) < 0 or \
           vol4/(1./6) > 1:
            v_point[i] = -1
        else:
            v_point[i] = v1*(vol1/(1./6)) + v2*(vol2/(1./6)) + \
                         v3*(vol3/(1./6)) + v4*(vol4/(1./6))

    return v_point


@jit([nb.float32[:](nb.float32[:], nb.float32[:,:]),
      nb.float64[:](nb.float64[:], nb.float64[:,:])],
     nopython=True)
def distance_3d(xn, x):
    """Distance of nodes to points"""
    return sqrt((xn[0]-x[:,0])**2 + (xn[1]-x[:,1])**2 + (xn[2]-x[:,2])**2)


@jit([nb.float32[:](nb.float32[:], nb.float32[:,:], nb.int32),
      nb.float64[:](nb.float64[:], nb.float64[:,:], nb.int32)],
     nopython=True)
def weight_3d(xn, x, p):
    """Weight of nodes to point"""
    return 1 / distance_3d(xn, x)**p


@jit([nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
                    nb.float32[:,:],
                    nb.float32, nb.float32, nb.float32, nb.float32),
      nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    nb.float64[:,:],
                    nb.float64, nb.float64, nb.float64, nb.float64)],
     nopython=True)
def nearest_3d(p1, p2, p3, p4, point, v1, v2, v3, v4):
    """JIT optimized nearest neighbor interpolation for 2D element"""
    values = (v1, v2, v3, v4)

    dist = np.empty((4, point.shape[0]), dtype=point.dtype)
    dist[0] = distance_3d(p1, point)
    dist[1] = distance_3d(p2, point)
    dist[2] = distance_3d(p3, point)
    dist[3] = distance_3d(p4, point)

    nearest = np.empty(point.shape[0], dtype=point.dtype)
    for j in range(dist.shape[1]):
        nearest[j] = values[np.argmin(dist[:,j])]

    return nearest


@jit([nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
                    nb.float32[:,:],
                    nb.float32, nb.float32, nb.float32, nb.float32,
                    nb.int32),
      nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    nb.float64[:,:],
                    nb.float64, nb.float64, nb.float64, nb.float64,
                    nb.int32)],
     nopython=True)
def idw_simple_3d(p1, p2, p3, p4, point, v1, v2, v3, v4, power):
    """Simple inverse distance weighting function"""
    v_point = (v1*weight_3d(p1, point, power) +
               v2*weight_3d(p2, point, power) +
               v3*weight_3d(p3, point, power) +
               v4*weight_3d(p4, point, power)) / \
              (weight_3d(p1, point, power) +
               weight_3d(p2, point, power) +
               weight_3d(p3, point, power) +
               weight_3d(p4, point, power))
    return v_point


jit_functions = {}

jit_functions['3d'] = {}
jit_functions['3d']['linear'] = linear_3d
jit_functions['3d']['idw'] = idw_simple_3d
jit_functions['3d']['nearest'] = nearest_3d

jit_functions['2d']= {}
jit_functions['2d']['cpu_linear'] = jit_setup.linear_cpu_setup
jit_functions['2d']['cpu_idw'] = jit_setup.idw_cpu_setup
jit_functions['2d']['cpu_nearest'] = jit_setup.nearest_cpu_setup
jit_functions['2d']['gpu_linear'] = jit_setup.linear_gpu_setup
jit_functions['2d']['gpu_idw'] = jit_setup.idw_gpu_setup
jit_functions['2d']['gpu_nearest'] = jit_setup.nearest_gpu_setup
