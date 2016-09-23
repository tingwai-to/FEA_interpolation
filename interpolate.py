import numpy as np
import numpy.linalg as npla
# import matplotlib.pyplot as plt
# import sys

def interpolate3D(p1, p2, p3, p4, p5, v1, v2, v3, v4):
    J = np.array([[p2[0]-p1[0], p3[0]-p1[0], p4[0]-p1[0]],
                  [p2[1]-p1[1], p3[1]-p1[1], p4[1]-p1[1]],
                  [p2[2]-p1[2], p3[2]-p1[2], p4[2]-p1[2]]])
    J = npla.inv(J)

    # ref1 = J.dot(np.array([ p1[0]-p1[0], p1[1]-p1[1], p1[2]-p1[2] ]))
    # ref2 = J.dot(np.array([ p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2] ]))
    # ref3 = J.dot(np.array([ p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2] ]))
    # ref4 = J.dot(np.array([ p4[0]-p1[0], p4[1]-p1[1], p4[2]-p1[2] ]))
    ref5 = J.dot(np.array([ p5[0]-p1[0], p5[1]-p1[1], p5[2]-p1[2] ]))

    tot_vol = 1./6  # volume of trirectangular tetrahedron
    # Volume of tetrahedron = 1/3 * base_area * height
    vol2 = (1./3)*(1./2)*ref5[0]
    vol3 = (1./3)*(1./2)*ref5[1]
    vol4 = (1./3)*(1./2)*ref5[2]
    vol1 = tot_vol - vol2 -vol3 - vol4

    v5 = v1*(vol1/tot_vol) + v2*(vol2/tot_vol) + v3*(vol3/tot_vol) + v4*(vol4/tot_vol)

    print v5
    return v5

def interpolate2D(p1, p2, p3, p4, v1, v2, v3):
    trans = np.array([[ p2[0]-p1[0], p3[0]-p1[0] ],
                      [ p2[1]-p1[1], p3[1]-p1[1] ]])
    trans = npla.inv(trans)

    # ref1 = trans.dot(np.array([ p1[0]-p1[0], p1[1]-p1[1] ]))
    # ref2 = trans.dot(np.array([ p2[0]-p1[0], p2[1]-p1[1] ]))
    # ref3 = trans.dot(np.array([ p3[0]-p1[0], p3[1]-p1[1] ]))
    ref4 = trans.dot(np.array([ p4[0]-p1[0], p4[1]-p1[1] ]))

    tot_area = 0.5  # area of right unit triangle
    area2 = 0.5*1*ref4[0]
    area3 = 0.5*1*ref4[1]
    area1 = tot_area - area2 - area3

    v4 = v1*(area1/tot_area) + v2*(area2/tot_area) + v3*(area3/tot_area)
    # Anywhere outside of reference triangle, set to zero

    print v4
    return v4


# 2D sample points
p1 = np.array([2,2])
p2 = np.array([4,3])
p3 = np.array([1,4])
p4 = np.array([2.5, 2.5])  # Value to be interpolated
v1 = 1
v2 = 2
v3 = 12

interpolate2D(p1, p2, p3, p4, v1, v2, v3)

# 3D sample points
p1 = np.array([2,2,1])
p2 = np.array([4,3,5])
p3 = np.array([1,4,3])
p4 = np.array([6,5,3])
p5 = np.array([3,3.5,3]) # Value to be interpolated
v1 = 1
v2 = 3
v3 = 12
v4 = 20

interpolate3D(p1, p2, p3, p4, p5, v1, v2, v3, v4)

# sys.exit()

x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))

N = 128
buff = np.zeros((N,N), dtype="f8")

x, y = np.mgrid[x_min:x_max:N*1j, y_min:y_max:N*1j]
p4 = np.array([x.ravel(), y.ravel()])
buff = interpolate2D(p1, p2, p3, p4, v1, v2, v3).reshape((N, N)).T
print buff

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

plt.imshow(buff, extent=[x_min, x_max, y_min, y_max], origin='lower')
plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.colorbar()
plt.savefig("output.png")
