#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sys import path
path.append('../')

from src import nnls
from src.operators import SubFFT

#
# Parameters
#
x_dim = 10000
y_dim = 8810
s = 50
print(f'Dimension = ({x_dim},{y_dim}), sparsity = {s}')

# Estimate of the number of measurements
target_m = s * np.log(x_dim) * np.log(s)**2

#
# Data
#
np.random.seed(1)

x_ground = np.zeros(x_dim)
x_ground[:s] = np.abs(np.random.randn(s))

idxs = np.random.choice(x_dim, y_dim, replace=False)
Bop = SubFFT(x_dim, idxs)
y = Bop @ x_ground

#
# Initialization
#
results = {}
eta = 2e-1
alphas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for alpha in alphas:
    results[alpha] = nnls.gd(Bop, y, alpha, eta, 10000, save_x=False)

for alpha, res in results.items():
    print(f'alpha = {alpha:.0e}, ground_error = {np.linalg.norm(res[0] - x_ground):.2e}')

# Convergence rates for different alpha
plt.figure()
for alpha, res in results.items():
    plt.semilogy(res[-1], label=f'{alpha:.0e}')
plt.title(f'GD, 2 layers, {y_dim}-Subsampled {x_dim}-dim FFT\nEffect of initialization')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend(title=r'$\alpha$')
plt.grid()
plt.savefig('figures/FFT_large_init.pdf')

# Support identification
plt.figure()
data = [np.log10(np.abs(results[alpha][0])) for alpha in [1e-2, 1e-3, 1e-6]]
plt.hist(data, bins=11, label=['1e-2', '1e-3', '1e-6'])
plt.yscale('log')
plt.xlabel('Magnitude of coordinates')
plt.title('Support identification in FFT')
plt.legend(title=r'$\alpha$')
# Change x tick labels
xt = np.arange(-15, 1, 3)
plt.xticks(xt, ['$10^{' + str(i) + '}$' for i in xt])
plt.savefig('figures/FFT_large_support.pdf')

# Effect of initialization on sparse recovery
fig, axs = plt.subplots(ncols=5, figsize=(15, 4))
for alpha, ax in zip(alphas, axs):
    ax.hist(np.log10(np.abs(results[alpha][0] - x_ground) + 1e-16), bins=100)
    ax.set_title(f'\\alpha = {alpha:.0e}')
    ax.set_yscale('log')
fig.suptitle('Histogram of (log10) sparse recovery errors\nEffect of initialization')
fig.savefig('figures/FFT_large_hist_error.pdf')
