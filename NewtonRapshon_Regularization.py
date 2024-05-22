from GradientBasedOptimization import GradientBasedOptimization as GBO

from numpy import array, dot, eye, float64
from numpy.linalg import inv
from time import time



class NewtonRapshonMethod_R(GBO):
    def __init__(self, function:str, variables:tuple[str], x0:array, epsilon=1e-4):
        super().__init__(function, variables, x0, epsilon)
    

    def optimize(self):
        start = time()

        self.aborted = False

        
        gradient_matrix = self.gradient_matrix()
        hessian_matrix = self.hessian_matrix(gradient_matrix)

        k = 0

        lambda_reg = 1e-4  # Düzenleme terimi

        while True:
            gm_xk = self.matrix_at_xk(gradient_matrix)
            if all(gm_xk <= self.epsilon):
                print(f'g_k <= epsilon')
                break 
            
            hm_xk = self.matrix_at_xk(hessian_matrix).astype(float64)  # Veri tipini float64 yapıyoruz
        
            
            # Düzenleme terimini Jacobi matrisine ekliyoruz
            hm_xk_inv = inv(hm_xk + lambda_reg * eye(hm_xk.shape[0]))

            if k == 0:
                print(f'x{k} = \n{self.xk}')
                print(f'\nf(x{k}) = {self.func_at_xk()}\n')

            xk_old = self.xk

            # The Newton Raphson Formula
            self.xk = self.xk - dot(hm_xk_inv, gm_xk)

            for i in range(self.xk.shape[0]):
                self.xk[i][0] = round(self.xk[i][0], 6)
            
            k += 1
            print(f'x{k} = \n{self.xk}')
            print(f'f(x{k}) = {self.func_at_xk()}\n')

            delta_matrix = abs(self.xk - xk_old)
            max_delta = float('-inf')
            for i in delta_matrix:
                if i > max_delta:
                    max_delta = i
            
            print('max_delta: ', max_delta, '\n')
            if max_delta < self.epsilon:
                print('max_delta < self.epsilon!')
                break

            if k == 1000: 
                print("Algorithm could not find the solution in {} iteration. So it was aborted.".format(k))
                self.aborted = True
                break

        end = time()
        if self.aborted == False:
            print(f'Optimization is done.\nElapsed time: {round(abs(start-end), 3)}\nTotal Iteration: {k}')
        
        else:
            print(f'Optimization had been run for {round(abs(start-end), 3)} second until it was aborted.')
            print(f'Current gradient matrix in x{k}:\n', gm_xk, '\n')





                


                

                
                

