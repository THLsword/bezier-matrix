from sympy import symbols, Matrix, diff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # Define symbols
# x, y = symbols('x y')

# # Define matrix
# A = Matrix([[x**2, x*y], [x*y, y**2]])

# # Take partial derivative of matrix with respect to x
# dAdx = A.applyfunc(lambda a: diff(a, x))

# # Print result
# print(dAdx)

A = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    [-7, 7, 0, 0, 0, 0, 0, 0],
    [21, -42, 21, 0, 0, 0, 0, 0],
    [-35, 105, -105, 35, 0, 0, 0, 0],
    [35, -140, 210, -140, 35, 0, 0, 0],
    [-21, 105, -210, 210, -105, 21, 0, 0],
    [7, -42, 105, -140, 105, -42, 7, 0],
    [-1, 7, -21, 35, -35, 21, -7, 1]
    ])
# t = np.power(t, np.array([0, 1, 2, 3, 4, 5, 6, 7]))  # [n_samples, 4]
# points = t @ A @ params  # [..., n_samples, 3]

# sides = [patches[:8, :], patches[7:15, :],
#              patches[14:22, :], patches[[21, 22, 23, 24, 25, 26, 27, 0], :]] 
# corners = [patches[[0], :], patches[[7], :],
#             patches[[21], :], patches[[14], :]]
        
# B = corners[0] * (1-s) * (1-t) + corners[1] * s * (1-t) + corners[2] * (1-s) * t + corners[3] * s * t 
# Lc = (s @ A @ sides[0])*(1-t) + ((1-s) @ A @ sides[2])*(t)
# Ld = (t @ A @ sides[1])*(s) + ((1-t) @ A @ sides[3])*(1-s)
# P = Lc + Ld - B
t = np.linspace(0, 1, 5).reshape((5, 1))
t = np.power(t, np.array([0, 1]))  # [n_samples, 4]
# TA = t @ A
s = np.linspace(0, 1, 5).reshape((5, 1))
# TA = TA * s
print(t)
print(s.shape)
print(t * s)

Ss, St = symbols('s t')
St =  Matrix(t)
St = Matrix([St**i for i in range(8)])
print(St.shape)
print(St)

s, t = symbols('s t')
s = Matrix([s]*144)
t = Matrix([t]*144)

# sides[0]).shape = (8,3)
Lc = (s @ A @ sides[0])*(1-t)

# Derivative of Lc w.r.t s
Lc_s = Lc.jacobian(s)

t = Matrix([t**i for i in range(8)])

