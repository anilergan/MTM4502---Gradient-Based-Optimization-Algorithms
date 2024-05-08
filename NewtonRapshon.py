from numpy import array, ndarray, append, dot
from numpy.linalg import inv
from sympy import sympify, Symbol, diff
from time import time



class NewtonRapshonMethod:
    def __init__(self, function:str, variables:tuple[str], x0:array, epsilon=1e-4):
        """
        While function is defining, power notations must be writen with double ** like: x1**2
        """
        
        # PARAMETER CHECK ----------------------------------
        # Check if 'function' parameter is a string
        if not isinstance(function, str):
            raise TypeError("The 'function' parameter must be a string.")
        
        # Check if 'variables' parameter is a tuple of strings
        if not isinstance(variables, tuple) or not all(isinstance(var, str) for var in variables):
            raise TypeError("The 'variables' parameter must be a tuple of strings.")
        
        # Check if 'x0' parameter is a numpy array
        if not isinstance(x0, ndarray):
            raise TypeError("The 'x0' parameter must be a numpy array.")
        # -------------------------------------------------
        
        self.function = sympify(function)

        self.V = []
        for var in variables:
            self.V.append(Symbol(var))
        self.V = tuple(self.V)

        self.xk = x0 # Initialize the xk with x0

        self.epsilon = epsilon

        


    def optimize(self):
        start = time()

        k = 0
        while True:

            
            

            gradient_matrix = self.gradient_matrix()
            hessian_matrix = self.hessian_matrix(gradient_matrix)

            gm_xk = self.matrix_at_xk(gradient_matrix)
            
            hm_xk = self.matrix_at_xk(hessian_matrix)
            hm_xk_inv = inv(hm_xk)
            
            if k == 0:
                print(f'x{k} = \n{self.xk}')
                print(f'f(x{k}) = {self.func_at_xk(self.function, self.xk)}\n')

            xk_old = self.xk

            # The Newton Raphson Formula
            self.xk = self.xk - dot(hm_xk_inv, gm_xk)


            for i in range(self.xk.shape[0]):
                self.xk[i][0] = round(self.xk[i][0], 6)
            
            k += 1
            print(f'x{k} = \n{self.xk}')
            print(f'f(x{k}) = {self.func_at_xk(self.function, self.xk)}\n')

            delta_matrix = abs(self.xk - xk_old)
            max_delta = float('-inf')
            for i in delta_matrix:
                if i > max_delta:
                    max_delta = i
            
            print('max_delta: ', max_delta, '\n')
            if max_delta < self.epsilon:
                print('max_delta < self.epsilon!')
                break
            
            

        end = time()

        print(f'Optimization is done. Elapsed time: {round(abs(start-end), 3)}')


    def gradient_matrix(self):

        gradient_matrix = array([])
        for v in self.V:
            derivative = diff(self.function, v)
            gradient_matrix = append(gradient_matrix, derivative)

        gradient_matrix = gradient_matrix.reshape(-1,1)

        return gradient_matrix
 

    def hessian_matrix(self, gm):
        hessian_matrix = array([])
        for vector in gm:
            for v in self.V:
                if v in vector[0].free_symbols:
                    derivative = diff(vector[0], v)
                    try: 
                        derivative = float("{:.6f}".format(derivative))
                        if derivative.is_integer():
                            derivative = int(derivative)
                    except TypeError as Error:
                        pass
            
                else: 
                    derivative = 0

                hessian_matrix = append(hessian_matrix, derivative)

        hessian_matrix = hessian_matrix.reshape(gm.shape[0], len(self.V))

        return hessian_matrix


    def matrix_at_xk(self, matrix):
        
        hm_xk = array([])
        for hm_item  in list(matrix.flatten()):
            
            if not isinstance(hm_item,int) or isinstance(hm_item, float):
                variable_values_at_xk = {}
                for i in range(len(self.V)):
                    variable_values_at_xk[self.V[i]] = self.xk[(i, 0)]
                
                hm_item = hm_item.subs(variable_values_at_xk)

                try: 
                    hm_item = float("{:.6f}".format(hm_item))
                    if hm_item.is_integer():
                        hm_item = int(hm_item)
                except TypeError as Error:
                    pass
            
            hm_xk = append(hm_xk, hm_item)
        
        return hm_xk.reshape(matrix.shape)


    def func_at_xk(self, func, xk:array):
        variable_values_at_xk = {}
        for i in range(len(self.V)):
            variable_values_at_xk[self.V[i]] = xk[(i, 0)]
                
        return float('{:.6f}'.format(func.subs(variable_values_at_xk)))


                


                

                
                

