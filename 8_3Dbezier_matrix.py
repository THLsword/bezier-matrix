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

# Generate random control points
P0 = np.array([1, 1, 0])
P1 = np.array([1, 3, 1])
P2 = np.array([1, 2, 2])
P3 = np.array([1, 2, 3])
P4 = np.array([6, 5, 4])
P5 = np.array([7, 7, 5])
P6 = np.array([8, 8, 4])
P7 = np.array([9, 4, 2])
params = np.stack([P0, P1, P2, P3, P4, P5, P6, P7], axis=0)

# Sample points on the Bezier curve
t = np.linspace(0, 1, 100).reshape((100, 1))
curve = bezier_sample(t, params)

# Plot the curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])
ax.scatter(params[:, 0], params[:, 1], params[:, 2], c='r')
plt.show()