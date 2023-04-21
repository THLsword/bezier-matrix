
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

A = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    [-7, 7, 0, 0, 0, 0, 0, 0],
    [21, -42, 21, 0, 0, 0, 0, 0],
    [-35, 105, -105, 35, 0, 0, 0, 0],
    [35, -140, 210, -140, 35, 0, 0, 0],
    [-21, 105, -210, 210, -105, 21, 0, 0],
    [7, -42, 105, -140, 105, -42, 7, 0],
    [-1, 7, -21, 35, -35, 21, -7, 1]
    ])

P0 = np.array([1*2, 1, 1])
P1 = np.array([2*2, 1.2, 1.2])
P2 = np.array([3*2, 1, 1.3])
P3 = np.array([4*2, 1.2, 1.4])
P4 = np.array([5*2, 1, 1.3])
P5 = np.array([6*2, 1.2, 1.2])
P6 = np.array([8*2, 1, 1])
P7 = np.array([9*2, 1.2, 1])

P8 = np.array([9.5*2, 1, 1.1])
P9 = np.array([9*2, 2, 1.5])
P10 = np.array([9.5*2, 3, 0.8])
P11 = np.array([9*2, 4, 0.6])
P12 = np.array([9.5*2, 5, 0.9])
P13 = np.array([9*2, 6, 1])
P14 = np.array([9.5*2, 7, 1.2])

P15 = np.array([9*2, 7.3, 1.2])
P16 = np.array([8*2, 7, 1.3])
P17 = np.array([7*2, 7.3, 0.8])
P18 = np.array([6*2, 7, 0.5])
P19 = np.array([5*2, 7.3, 0.5])
P20 = np.array([4*2, 7, 0.8])
P21 = np.array([2*2, 7.3, 1])

P22 = np.array([2*2, 7, 1])
P23 = np.array([2*2, 6, 1.3])
P24 = np.array([2*2, 5, 1.3])
P25 = np.array([2*2, 4, 0.7])
P26 = np.array([2*2, 3, 0.7])
P27 = np.array([2*2, 2, 1])

patches = np.stack([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27], axis=0)

sides = [patches[:8, :], patches[7:15, :],patches[14:22, :], patches[[21, 22, 23, 24, 25, 26, 27, 0], :]] 
corners = [patches[[0], :], patches[[7], :],patches[[21], :], patches[[14], :]]

t = np.random.rand(100).reshape((100, 1))
s = np.random.rand(100).reshape((100, 1))

B = corners[0] * (1-s) * (1-t) + corners[1] * s * (1-t) + \
        corners[2] * (1-s) * t + corners[3] * s * t  # [..., n_samples, 3]
Lc = np.concatenate((s**0,s**1,s**2,s**3,s**4,s**5,s**6,s**7), axis=1) @ A @ sides[0] * (1-t) +\
    np.concatenate(((1-s)**0,(1-s)**1,(1-s)**2,(1-s)**3,(1-s)**4,(1-s)**5,(1-s)**6,(1-s)**7), axis=1) @ A @ sides[2] * t
Ld = np.concatenate((t**0,t**1,t**2,t**3,t**4,t**5,t**6,t**7), axis=1) @ A @ sides[1] * s +\
    np.concatenate(((1-t)**0,(1-t)**1,(1-t)**2,(1-t)**3,(1-t)**4,(1-t)**5,(1-t)**6,(1-t)**7), axis=1) @ A @ sides[3] * (1-s)

coons_point = Lc + Ld - B

dLc_dt = np.concatenate((np.zeros_like(t),np.ones_like(t),2*t,3*t**2,4*t**3,5*t**4,6*t**5,7*t**6), axis=1) @ A @ sides[0] * (-1) +\
    np.concatenate((np.ones_like(t),np.zeros_like(t),2*(1-t),3*(1-t)**2,4*(1-t)**3,5*(1-t)**4,6*(1-t)**5,7*(1-t)**6), axis=1) @ A @ sides[2] * (1-s)
dLd_dt = np.concatenate((s**0,s**1,s**2,s**3,s**4,s**5,s**6,s**7), axis=1) @ A @ sides[1] * (1-t) +\
    np.concatenate((-s**0,-s**1,-s**2,-s**3,-s**4,-s**5,-s**6,-s**7), axis=1) @ A @ sides[3] * (1-s)

dcoons_point_dt = dLc_dt + dLd_dt


# partial derivative of coons_point w.r.t s
dLc_ds = np.concatenate((np.zeros_like(t),np.ones_like(t),2*s,3*s**2,4*s**3,5*s**4,6*s**5,7*s**6), axis=1) @ A @ sides[0] * (1-t) +\
    np.concatenate((-np.ones_like(t),np.zeros_like(t),-2*s,-3*s**2,-4*s**3,-5*s**4,-6*s**5,-7*s**6), axis=1) @ A @ sides[2] * t
dLd_ds = np.concatenate((t**0,t**1,t**2,t**3,t**4,t**5,t**6,t**7), axis=1) @ A @ sides[1] +\
    np.concatenate((-t**0,-t**1,-t**2,-t**3,-t**4,-t**5,-t**6,-t**7), axis=1) @ A @ sides[3] * (1-s)

dcoons_point_ds = dLc_ds + dLd_ds
normal = np.cross(dcoons_point_dt, dcoons_point_ds)
normal /= np.linalg.norm(normal)
end_points = coons_point+normal
print("end_points.shape: ", end_points.shape)
print(dcoons_point_dt.shape)
print(dcoons_point_ds.shape)
print(normal.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], c='b')

ax.scatter(coons_point[:, 0], coons_point[:, 1], coons_point[:, 2], c='r')
plt.show()