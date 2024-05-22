from GradientBasedOptimization import GradientBasedOptimization as GBO
from time import time
from numpy import dot, array, all

class ConjugateGradientAlgorithm(GBO):
    def __init__(self, function:str, variables:tuple[str], x0:array, epsilon=1e-4):
        super().__init__(function, variables, x0, epsilon)
    

    def optimize_with_particaular_beta(self, beta_func):
        start = time()
        k = 0

        self.aborted = False


        g = self.gradient_matrix()
        Q = self.hessian_matrix(g)

        g_k = self.matrix_at_xk(g)

        if all(g_k == 0):
                print("Gradient Matrix is zero already at the initial point.")
                return
        else: 
            d_k = -g_k

        while True:

            Q_k = self.matrix_at_xk(Q)
            alfa_k = -(dot(g_k.T, d_k)/dot(dot(d_k.T, Q_k), d_k))

            if k == 0:
                print(f'x{k} = \n{self.xk}')
                print(f'\nf(x{k}) = {self.func_at_xk()}\n')
                print(f'g{k} =\n{g_k}\n')
                print('-'*50)


            self.xk = self.xk + alfa_k * d_k
            k += 1
            print(f'x{k} = \n{self.xk}')
            print(f'\nf(x{k}) = {self.func_at_xk()}\n')
            
            g_k_old = g_k
            g_k =  self.matrix_at_xk(g)
            print(f'g{k} =\n{g_k}\n')

            if all(g_k <= self.epsilon):
                print(f'g_k <= epsilon')
                break   

            if all(abs(g_k - g_k_old) <= self.epsilon):
                print(f'|g(k) - g(k-1| <= epsilon')
                break  

            print('-'*50)    

            beta_k = beta_func(g_k_old, g_k, Q_k, d_k)

            d_k = -g_k + beta_k * d_k

            if k == 1000: 
                print("Algorithm could not find the solution in {} iteration. So it was aborted.".format(k))
                self.aborted = True
                break

        end = time()
        if self.aborted == False:
            print(f'Optimization is done.\nElapsed time: {round(abs(start-end), 3)}\nTotal Iteration: {k}')
        
        else:
            print(f'Optimization had been run for {round(abs(start-end), 3)} second until it was aborted.')
            print(f'Current gradient matrix in x{k}:\n', g_k, '\n')

    


