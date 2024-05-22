from ConjugateGradientAlgorithm import ConjugateGradientAlgorithm as CGA

from time import time
from numpy import dot, array, all

class PolakRibiere(CGA):
    def __init__(self, function:str, variables:tuple[str], x0:array, epsilon=1e-4):
        super().__init__(function, variables, x0, epsilon)
    
    def beta_func(self, g_k_old, g_k, Q_k, d_k):
        beta_k = dot(g_k.T, (g_k - g_k_old)) / dot(g_k_old.T, g_k_old)
        return beta_k
    
    def optimize(self):
        self.optimize_with_particaular_beta(self.beta_func)