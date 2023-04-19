import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    [-7, 7, 0, 0, 0, 0, 0, 0],
    [21, -42, 21, 0, 0, 0, 0, 0],
    [-35, 105, -105, 35, 0, 0, 0, 0],
    [35, -140, 210, -140, 35, 0, 0, 0],
    [-21, 105, -210, 210, -105, 21, 0, 0],
    [7, -42, 105, -140, 105, -42, 7, 0],
    [-1, 7, -21, 35, -35, 21, -7, 1]
    ])
    t = np.power(t, np.array([0, 1, 2, 3, 4, 5, 6, 7]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def coons_points(s, t, patches):
    """Sample points from Coons patch defined by params at s, t values.

    params -- [..., 12, 3]
    """
    # sides = [patches[..., :4, :], patches[..., 3:7, :],
    #          patches[..., 6:10, :], patches[..., [9, 10, 11, 0], :]]
    # corners = [patches[..., [0], :], patches[..., [3], :],
    #            patches[..., [9], :], patches[..., [6], :]]

    sides = [patches[:8, :], patches[7:15, :],
             patches[14:22, :], patches[[21, 22, 23, 24, 25, 26, 27, 0], :]] 
    corners = [patches[[0], :], patches[[7], :],
               patches[[21], :], patches[[14], :]]
           
    B = corners[0] * (1-s) * (1-t) + corners[1] * s * (1-t) + \
        corners[2] * (1-s) * t + corners[3] * s * t  # [..., n_samples, 3]

    Lc = bezier_sample(s, sides[0]) * (1-t) + bezier_sample(1-s, sides[2]) * t
    Ld = bezier_sample(t, sides[1]) * s + bezier_sample(1-t, sides[3]) * (1-s)
    
    return Lc + Ld - B

# Generate random control points
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

params = np.stack([P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27], axis=0)

# Sample points on the Bezier curve
t = np.linspace(0, 1, 100).reshape((100, 1))
s = np.linspace(0, 1, 100).reshape((100, 1))

# curve = bezier_sample(t, params)
curve1 = coons_points(np.linspace(0, 0, 100).reshape((100, 1)),np.linspace(0, 1, 100).reshape((100, 1)),params)
curve2 = coons_points(np.linspace(0, 1, 100).reshape((100, 1)),np.linspace(1, 1, 100).reshape((100, 1)),params)
curve3 = coons_points(np.linspace(1, 1, 100).reshape((100, 1)),np.linspace(0, 1, 100).reshape((100, 1)),params)
curve4 = coons_points(np.linspace(0, 1, 100).reshape((100, 1)),np.linspace(0, 0, 100).reshape((100, 1)),params)

curve5 = coons_points(np.linspace(0, 1, 100).reshape((100, 1)),np.linspace(0.25, 0.25, 100).reshape((100, 1)),params)
curve6 = coons_points(np.linspace(0, 1, 100).reshape((100, 1)),np.linspace(0.5, 0.5, 100).reshape((100, 1)),params)
curve7 = coons_points(np.linspace(0, 1, 100).reshape((100, 1)),np.linspace(0.75, 0.75, 100).reshape((100, 1)),params)

# print(curve.shape)

# Plot the curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve1[:, 0], curve1[:, 1], curve1[:, 2])
ax.plot(curve2[:, 0], curve2[:, 1], curve2[:, 2])
ax.plot(curve3[:, 0], curve3[:, 1], curve3[:, 2])
ax.plot(curve4[:, 0], curve4[:, 1], curve4[:, 2])

ax.plot(curve5[:, 0], curve5[:, 1], curve5[:, 2])
ax.plot(curve6[:, 0], curve6[:, 1], curve6[:, 2])
ax.plot(curve7[:, 0], curve7[:, 1], curve7[:, 2])



ax.scatter(params[:, 0], params[:, 1], params[:, 2], c='r')
plt.show()