from sympy import Symbol
from numpy import array

from NewtonRapshon import NewtonRapshonMethod as NR


f = "0.25*(x1**4) - 0.5*(x1**2) + 0.1*x1 + 0.5*(x2**2)"

x0 =array([[1], [-1]])

model = NR(
    function= f,
    variables=('x1', 'x2'), 
    x0=x0)

model.optimize()

