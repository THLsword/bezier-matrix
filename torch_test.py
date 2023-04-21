import numpy as np

# 定义控制点和基函数
P = np.array([
    [[0, 0, 0], [1, 0, 1], [2, 0, 0]],
    [[0, 1, 1], [1, 1, 0], [2, 1, 1]],
    [[0, 2, 0], [1, 2, 1], [2, 2, 0]],
    [[0, 3, 1], [1, 3, 0], [2, 3, 1]]
])

N = lambda u: np.array([(1 - u)**3, 3*u*(1 - u)**2, 3*u**2*(1 - u), u**3])

# 定义计算法向量的函数
def get_normal(u, v):
    Bu = np.array([N(u), 0, -N(u), 0])
    Bv = np.array([0, N(v), 0, -N(v)])
    Puv = np.dot(np.dot(Bu, P), np.dot(P.T, Bv.T))
    du, dv = np.dot(Bu, P), np.dot(Bv, P)
    n = np.cross(du, dv)
    return n / np.linalg.norm(n)

# 计算曲面上点的法向量
u = 0.5
v = 0.5
n = get_normal(u, v)

# 输出结果
print("在点 ({}, {}) 处的法向量为: {}".format(u, v, n))
