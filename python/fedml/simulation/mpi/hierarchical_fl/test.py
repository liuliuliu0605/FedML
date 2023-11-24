import numpy as np

x = np.array([1, 2, 3, 4])
p = np.array([1/6, 1/6, 1/3, 1/3])
# p = np.array([1/4, 1/4, 1/4, 1/4])
W = np.array(
    [[1/3, 1/3, 1/3,  0],
     [1/3, 1/3, 0,   1/3],
     [1/3, 0,   1/3, 1/3],
     [0,   1/3, 1/3, 1/3]])


W = np.array(
    [[1/4, 1/4, 1/4, 1/4],
     [1/4, 1/4, 1/4, 1/4],
     [1/4, 1/4, 1/4, 1/4],
     [1/4, 1/4, 1/4, 1/4]])

print(np.mean(x))
print(np.sum(x*p))

epsilon = min(p/4)
print(epsilon)

for t in range(10):
    rs = []
    for i in range(4):
        x_new = x[i] + np.sum((x - x[i]) * W[:, i])
        # x_new = x[i] + np.sum((x - x[i]) * W[:, i]) / p[i] * epsilon
        rs.append(x_new)
    x = rs
    print(rs)

