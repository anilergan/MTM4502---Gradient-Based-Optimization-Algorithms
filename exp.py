from sympy import symbols
import numpy as np
from numpy import e


# x0 = np.array([[1], [1]])

# teta = np.array([[-0.33333, -0.16666], [-0.16666, -0.33333]])

# gm = np.array([[2], [0]])


# old = np.array([[2], [0]])
# new = np.array([[2.003], [0.001]])


# x1 = x0 - np.dot(teta,gm)

# print(abs(old-new))

def define_the_function(x1 = 1, x2 = 10, x3 = 1, x4 = 5):
    f = None
    for i in range(1,11):
        t = 0.1 * i
        y = (e**(-t)) - 5*(e**(10*t))

        fi = ((x3*(e**(-t*x1)) - x4*(e**(-t*x2)) - y)**2)

        if i == 1:
            print('1. İterasyonuda Hesaplanan f:',fi)
            return

        
        # print(fi)
        if not f:
            f = fi
        else: 
            f += fi
    
    print(f)


define_the_function()


