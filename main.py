from sympy import Symbol
from numpy import array, e, random

from GradientBasedOptimization import GradientBasedOptimization as GBO

from NewtonRapshon import NewtonRapshonMethod as NR
from NewtonRapshon_Regularization import NewtonRapshonMethod_R as NR_R
from HestenesStiefel import HestenesStiefelMethod as HS
from PolakRibiere import PolakRibiere as PR
from FletcherReeves import FletcherReeves as FR


def define_the_function_1():
    f = ""
    for i in range(1,3):
        t = f"(0.1 * {i})"
        y = f"({e}**(-{t}) - 5*{e}**(-10*{t}))"
        fi = f"(((x3)*{e}**(-{t}*x1) - x4*{e}**(-{t}*x2) - {y})**2)"
        
        if not f:
            f = fi
        else: 
            f = f + ' + ' + fi
    
    return f



f = define_the_function_1()

print('Function:', f)



random.seed(1)
x0_1 = array(random.randn(4)).reshape(-1,1)
x0_2 = array(random.randn(4)).reshape(-1,1)
x0_3 = array(random.randn(4)).reshape(-1,1)


print("\nFirst Initial Point:", x0_1.flatten())
print("Second Initial Point:", x0_2.flatten())
print("Thirth Initial Point:", x0_3.flatten())
print('\n', '~'*70)


model = GBO(
    function= f,
    variables=('x1', 'x2', 'x3', 'x4'), 
    x0=x0_1)

solution_x = array([1,10,1,5]).reshape(-1,1)

solution_dic = {
    'x1' : solution_x.flatten()[0],
    'x2': solution_x.flatten()[1],
    'x3': solution_x.flatten()[2],
    'x4': solution_x.flatten()[3]
}


solution_value = model.func_at_particular_x(solution_dic)
print(solution_value)

model = NR(
    function= f,
    variables=('x1', 'x2', 'x3', 'x4'), 
    x0=x0_3)


model = NR_R(
    function= f,
    variables=('x1', 'x2', 'x3', 'x4'), 
    x0=x0_3)

model = HS(
    function= f,
    variables=('x1', 'x2', 'x3', 'x4'), 
    x0=x0_3)

model = PR(
    function= f,
    variables=('x1', 'x2', 'x3', 'x4'), 
    x0=x0_3)

model = FR(
    function= f,
    variables=('x1', 'x2', 'x3', 'x4'), 
    x0=x0_3)

model.optimize()

