
from numpy import array, append, ndarray
from sympy import sympify, Symbol, diff



class GradientBasedOptimization():
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


    def func_at_xk(self):
        variable_values_at_xk = {}
        for i in range(len(self.V)):
            variable_values_at_xk[self.V[i]] = self.xk[(i, 0)]
                
        return float('{:.6f}'.format(self.function.subs(variable_values_at_xk)))
    

    def func_at_particular_x(self, variable_values_at_x):
        return float('{:.6f}'.format(self.function.subs(variable_values_at_x)))
