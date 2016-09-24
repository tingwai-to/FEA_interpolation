import interpolate
import numpy as np


# 2D sample points
p1 = np.array([2,2])
p2 = np.array([4,3])
p3 = np.array([1,4])
p4 = np.array([2,2])  # Value to be interpolated
v1 = 1
v2 = 2
v3 = 3


# 3D sample points
# p1 = np.array([2,2,1])
# p2 = np.array([4,3,5])
# p3 = np.array([1,4,3])
# p4 = np.array([6,5,3])
# p5 = np.array([3,3.5,3]) # Value to be interpolated
# v1 = 1
# v2 = 2
# v3 = 3
# v4 = 4


x_min = min(_[0] for _ in (p1, p2, p3))
y_min = min(_[1] for _ in (p1, p2, p3))
x_max = max(_[0] for _ in (p1, p2, p3))
y_max = max(_[1] for _ in (p1, p2, p3))


N = 128
buff = np.zeros((N, N), dtype="f8")
for i, x in enumerate(np.linspace(x_min,x_max,N)):
    for j, y in enumerate(np.linspace(y_min,y_max,N)):
        buff[i][j] = interpolate.interpolate2d(p1,p2,p3,np.array([x,y]),v1,v2,v3)


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], '-k')
plt.imshow(buff.T, extent=[x_min, x_max, y_min, y_max], origin='lower')
plt.colorbar()
plt.savefig("output2D.png")
