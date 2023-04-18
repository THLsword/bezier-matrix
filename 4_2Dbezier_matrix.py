import numpy as np
import matplotlib.pyplot as plt

def cubic_bezier(t, P0, P1, P2, P3):
    """
    Computes the coordinates of the point on a cubic Bezier curve defined by control points P0, P1, P2, P3 at parameter t.
    """
    B = np.array([[1, -3, 3, -1],
                  [0, 3, -6, 3],
                  [0, 0, 3, -3],
                  [0, 0, 0, 1]])
    T = np.array([t**3, t**2, t, 1])
    P = np.array([P0, P1, P2, P3])
    return T @ B @ P

# Define control points
P0 = np.array([0, 0])
P1 = np.array([1, 1])
P2 = np.array([2, -1])
P3 = np.array([3, 0])

# Sample the curve
n_samples = 100
t = np.linspace(0, 1, n_samples)
curve = np.array([cubic_bezier(ti, P0, P1, P2, P3) for ti in t])

# Plot the curve and control points
plt.plot(curve[:,0], curve[:,1])
plt.plot([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], 'ro')
plt.axis('equal')
plt.show()