from sympy import symbols
import numpy as np


x0 = np.array([[1], [1]])

teta = np.array([[-0.33333, -0.16666], [-0.16666, -0.33333]])

gm = np.array([[2], [0]])


old = np.array([[2], [0]])
new = np.array([[2.003], [0.001]])


x1 = x0 - np.dot(teta,gm)

print(abs(old-new))



