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
y_dim = 20
s = 3
print(f'Dimension = ({x_dim},{y_dim}), sparsity = {s}')

#
# Data
#
np.random.seed(3)
A, x_ground, b = datagen.Data_Generate_Positive(x_dim, y_dim, s, 'normal', 'normal')

results = {}

#
# Gradient descent: Riemannian
#
alpha = 1e-1
eta = 1e-1
T = int(20e4)
Ls = [2,3]

gammas = [0, 0.25, 0.5, 0.75]
for L in Ls:
    print(f'GD, {L} layers: Step size = {eta}, Initial  = {alpha}')
    for gamma in gammas:
        start_time = time.time()
        results[(L, gamma)] = nnls.gd(A, b, alpha, eta, T, L=L, gamma=gamma, save_x=True, save_e=True, riemannian=True)
        gd_time = time.time() - start_time

        print(f'GD step {gamma:0.2f} Error = {results[(L, gamma)][3][-1]:.5}')
        print(f'GD step {gamma:0.2f} Time  = {gd_time:.5} s')

    plt.figure()
    for gamma in gammas:
        plt.semilogy(la.norm(results[(L, gamma)][1] - x_ground, axis=1), label=f'{gamma:0.2f}')
    plt.legend(title="Gamma")

    plt.title(f'GD {L} Layers: Distance along iterations')
    plt.xlabel('Iteration')
    plt.grid()
    plt.savefig(f'figures/gd_gamma_{L}L_{x_dim}x{y_dim}_distance.pdf')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axs:
        for gamma in gammas:
            ax.plot(results[(L, gamma)][3], label=f'{gamma:0.2f}')
        ax.legend(title='Gamma')
        ax.set_title(f'GD {L} Layers: Training Error')
        ax.set_xlabel('Iteration')

        ax.grid()
        ax.set_yscale('log')

    ax1, ax2 = axs
    ax2.set_xscale('log')
    plt.savefig(f'figures/gd_gamma_{L}L_{x_dim}x{y_dim}_error.pdf')

fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for L,ax in zip(Ls,axs):
    for (n,gamma) in enumerate(gammas):
        if L == 2:
            K = 1
            u = 20/(1 + gamma)
        else:
            K = 10
            u = 10*(1 + gamma)
        ax.loglog(results[(L, gamma)][3], label=f'{gamma:.2f}')
        ax.loglog([K, T], [u, u*(T/K)**(gamma-1)], linestyle='--', color=f'C{n}')
    ax.legend(title=r"$\gamma$")
    ax.set_title(f'{L} layers')
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel('Iteration')
axs[0].set_ylabel('Error')

fig.savefig(f'figures/gd_gamma_{x_dim}x{y_dim}_error.pdf')