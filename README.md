#FEA Interpolation

Interpolate values at points inside of elements for finite element analysis (FEA). Supports linear, inverse distance weighting, and nearest neighbor interpolation.

##Features
Standard (Non-JIT): Supports interpolating in n-dimensional elements (2D, 3D, etc.)
CPU: Just-in-time (JIT) optimized functions, supported in 2D and 3D
GPU: CUDA optimized functions, supported in 2D

##Usage
###2D Element Example
Setup:
```
from feainterp.node import Node as Node
from feainterp.element import Element as Element

# Coordinates of points and respective values
p1 = np.array([2, 2], dtype='f8')
p2 = np.array([4, 3], dtype='f8')
p3 = np.array([1, 4], dtype='f8')
v1 = 1.
v2 = 2.
v3 = 3.

# Create Node object
node1 = Node(p1, v1)
node2 = Node(p2, v2)
node3 = Node(p3, v3)

# Create Element object
triangle = Element([node1, node2, node3])
```
This creates an Element object named `triangle` with `node1`, `node2`, `node3`.

Interpolation:
```
# n x dim shaped array of coordinates
points = np.array([[2.3, 2.5],
                   [3.2, 3.0]], dtype='f8')

result = triangle.sample('linear', points)             # standard non-jit
result = triangle.sample('linear', points, jit='cpu')  # CPU JIT
result = triangle.sample('linear', points, jit='gpu')  # GPU JIT
```
This linearly interpolates values at `points` inside `triangle` using non-jit, CPU, or GPU. Replace `'linear'` with `'idw'` or `'nearest'` for inverse distance weighting and nearest neighbor interpolation respectively.

##Prerequisites
* numpy
* scipy
* numba
* CUDA

##Thank you
I want to thank Matthew Turk and the [Data Exploration Lab](https://dxl.ncsa.illinois.edu/people/) team at the National Center for Supercomputing Applications [(NCSA)](http://www.ncsa.illinois.edu/) for their continued support and help.
