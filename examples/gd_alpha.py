import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

from sys import path
path.append('../')

from src import nnls

import datagen

#
# Parameters
#
x_dim = 50
y_dim = 10
s = 3
print(f'Dimension = ({x_dim},{y_dim}), sparsity = {s}')

#
# Data
#
seed = 3
np.random.seed(seed)
A, x_ground, b = datagen.Data_Generate_Positive(x_dim, y_dim, s, 'normal', 'normal')

results = {}

#
# Other algorithms
#
others = ['CVXPY', 'LH', 'BP']

# CVXPy solutions
import cvxpy as cp

# Basis Pursuit
x = cp.Variable(x_dim)
objective = cp.Minimize(cp.norm(x,1))
constraints = [A@x == b]
prob = cp.Problem(objective, constraints)

start = time.time()
result = prob.solve()
end = time.time()

print('CVXPy   BP: L1 norm:', la.norm(x.value, 1), 'Time:', end-start)
results['BP'] = [x.value]

# NNLS
z = cp.Variable(x_dim)
objective = cp.Minimize(cp.quad_form(z, A.T@A) - 2*b.T@A@z)
constraints = [z >= 0]
prob = cp.Problem(objective, constraints)

start = time.time()
result = prob.solve()
end = time.time()

print('CVXPy NNLS: L1 norm:', la.norm(z.value, 1), 'Time:', end-start)
results['CVXPY'] = [z.value]

# LH
from scipy.optimize import nnls as lh

start = time.time()
x_nnls, err_nnls = lh(A,b)
end = time.time()

print('   LH NNLS: L1 norm:', la.norm(x_nnls, 1), 'Time:', end-start)
results['LH'] = [x_nnls]

# Save results to compare with the NNLS implementation
import pickle
with open(f'results/cvxpy_lh_{x_dim}x{y_dim}.pkl', 'wb') as f:
    pickle.dump({'x': x.value, 'z': z.value, 'x_nnls': x_nnls, 'seed': seed}, f)

#
# Reparameterized NNLS
#

# Riemannian Gradient descent
eta = 1e-1
T = int(5e5)
Ls = [2,3]

alphas = np.logspace(-3, -1, 5)
for L in Ls:
    print(f'GD, {L} layers: Step size = {eta}')
    for alpha in alphas:
        print(f'Initial  = {alpha}')
        results[(L, alpha)] = nnls.gd(A, b, alpha, eta, T, L=L, save_x=True, save_e=True, riemannian=True)

#
# Comparison graphs
#


# L1 norm
plt.figure()

for (L, marker) in zip(Ls, ['x', 'o']):
    norms = [la.norm(results[(L, alpha)][0],1) for alpha in alphas]
    plt.semilogx(alphas, norms, label=f'GD, {L}L', marker=marker)

for (other, color) in zip(others, ['salmon', 'brown', 'black']):
    plt.axhline(la.norm(results[other][0], 1), color=color, linestyle='--', label=other)

plt.legend()

plt.title('Implicit bias')
plt.xlabel('Magnitude of initialization')
plt.ylabel('$\ell^1$-norm of solution')
plt.savefig(f'figures/gd_IB_alpha_{x_dim}x{y_dim}.pdf')

# Error
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for (L, ax) in zip(Ls, axs):
    for alpha in alphas:
        ax.semilogy(results[(L, alpha)][3], label=f'{alpha:.2e}')
    ax.legend(title=r"$\alpha$")
    ax.set_title(f'GD, {L} layers')
    ax.set_ylim(1e-6, 1e1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
fig.savefig(f'figures/gd_IB_error_{x_dim}x{y_dim}.pdf')
