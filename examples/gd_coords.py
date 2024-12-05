import numpy as np
import matplotlib.pyplot as plt

from sys import path
path.append('../')

from src import nnls

import datagen

#
# Parameters
#
x_dim = 50
y_dim = 30
s = 3
print(f'Dimension = ({x_dim},{y_dim}), sparsity = {s}')

#
# Data
#
seed = 3
np.random.seed(seed)
A, x_ground, b = datagen.Data_Generate_Positive(x_dim, y_dim, s, 'normal', 'normal')

# Riemannian Gradient descent
alpha = 1e-2
eta = 1e-2
T = int(3e5)
Ls = [2,3]

results = {}
for L in Ls:
    print(f'GD, {L} layers: Step size = {eta}')
    results[(L, alpha)] = nnls.gd(A, b, alpha, eta, T, L=L, save_x=True, save_e=True, riemannian=True)

#
# Comparison graphs
#

ords = ['st', 'nd', 'rd']
styles = ['-', '--', ':']
for L, color in zip(Ls, ['C0', 'C1']):
    for j in range(3):
        plt.plot(np.abs(results[(L, alpha)][1][:,j] - x_ground[j]), label=f'GD-{L}L, {j+1}{ords[j]} coord', color=color, linestyle=styles[j])
plt.legend()
plt.title('Convergence of coordinates')
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.yscale('log')
plt.savefig('figures/gd_coords.pdf')
