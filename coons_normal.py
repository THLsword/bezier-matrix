# wirtten by bing:
# def normal(s, t, u, v, patches):
#     # Compute partial derivatives
#     du = np.array([1, 0])
#     dv = np.array([0, 1])
#     point = coons_points(u, v, patches)
#     x_u = du @ coons_points(u, v, patches)
#     x_v = dv @ coons_points(u, v, patches)
#     # Compute cross product
#     n = np.cross(x_u, x_v)
#     # Normalize
#     n /= np.linalg.norm(n)
#     return n


import numpy as np

# Define the Coons patch function
def coons_patch(u, v):
    # Define the control points
    p00 = np.array([0, 0, 0])
    p01 = np.array([0, 1, 0])
    p10 = np.array([1, 0, 0])
    p11 = np.array([1, 1, 1])
    
    # Calculate the blending functions
    f1 = (1 - u) * p00 + u * p10
    f2 = (1 - u) * p01 + u * p11
    g1 = (1 - v) * p00 + v * p01
    g2 = (1 - v) * p10 + v * p11
    
    # Calculate the surface point
    s = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + (1 - u) * v * p01 + u * v * p11
    
    # Calculate the partial derivatives
    du = (1 - v) * (f2 - f1) + v * (g2 - g1)
    dv = (1 - u) * (g2 - g1) + u * (f2 - f1)
    
    # Calculate the normal vector
    n = np.cross(du, dv)
    
    return n

# Call the function with the desired values of u and v
u = 0.5
v = 0.5
coons_patch(u, v)