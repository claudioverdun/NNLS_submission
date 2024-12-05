import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sys import path
path.append('../')

from src import nnls

import datagen

#
# Parameters
#
x_dim = 50
y_dim = 20
s = 3
print('(x,y) Dimension = ({},{})'.format(x_dim,y_dim))

#
# Data
#
np.random.seed(42)
A, x_ground, b = datagen.Data_Generate_Positive(x_dim, y_dim, s, 'normal', 'normal')


#
# ODEs
#
def R(t, v):
    x = v[:x_dim]
    y = v[x_dim:]
    gradloss = A.T @ (A @ x - b)
    return np.concatenate([2/t*(y - x), -t/2 * y * gradloss])

#
# Solution to the ODE
#
I = [1e-3, 1e3]
alpha = 1e-2

sol = solve_ivp(R, I, alpha**2 * np.ones(2*x_dim), rtol=1e-6, atol=1e-8)

# 
# Convergence graph
#
T = I[-1]
err = la.norm((A @ sol.y[:x_dim,:]).T - b, axis=1)

plt.figure()
plt.loglog(sol.t, err**2, label='ODE')
plt.loglog([7, T], [1, 1*(T/7)**(-2)], ':', label='$k^{-2}$')
plt.loglog([7, T], [1, 1*(T/7)**(-3)], '--' , label='$k^{-3}$')
plt.title(f'ODE approach, dim(A) = {y_dim}x{x_dim}')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.savefig(f'figures/ODE_convergence_{y_dim}x{x_dim}.pdf')


#
# Accelerated algorithms
# 
alpha = 1e-3
eta = 1e-2
T = 18000

results_amd = nnls.gd_2layers_nesterov_mirror(A, b, alpha, eta, T)
results_nest = nnls.gd_2layers_nesterov(A, b, np.sqrt(alpha), eta, T, restart=False)

# Convergence graphs
plt.figure()
plt.loglog(results_amd[-1]**2, label=r'$\ell(\tilde{x})$')
plt.loglog(results_nest[-1]**2, label=r'$\mathcal{L}(x)$')
plt.plot([60, T], [2, 2*(T/60)**(-3)], '--', label='$k^{-3}$')
plt.plot([60, T], [2, 2*(T/60)**(-4)], '--', label='$k^{-4}$')
plt.legend()
plt.title('Acceleration')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.savefig(f'figures/Accel_GD_MD_{y_dim}x{x_dim}.pdf')
