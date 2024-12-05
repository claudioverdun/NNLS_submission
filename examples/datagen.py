#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from scipy.stats import bernoulli


def hilbert(n, m=0):
    if n < 1 or m < 0:
        raise ValueError("Matrix size must be one or greater")
    elif n == 1 and (m == 0 or m == 1):
        return np.array([[1]])
    elif m == 0:
        m = n

    v = np.arange(1, n + 1) + np.arange(0, m)[:, np.newaxis]
    return 1. / v

def random_vector(n, s, mode='normal', normalize=True, permute=False):
    """
    Generate a random vector of dimension `n` with `s` non-zero entries.
    `mode` can be 'normal', 'uniform', 'poly_decay', 'exp_decay',
    'small_normal', 'large_normal', and selects the distribution of the
    non-zero entries.
    """
    x = np.zeros(n)
    if mode == 'normal':
        x[:s] = np.random.normal(0, 1, s)
    elif mode == 'uniform':
        x[:s] = np.random.rand(s)
    elif mode == 'poly_decay':
        x[:s] = 3 ** (-np.arange(s))
    elif mode == 'exp_decay':
        x[:s] = np.exp(-np.arange(s))
    elif mode == 'small_normal':
        # mean of each entry is 1; used with s=n
        x[:s] = np.random.normal(1, 1, s)
    elif mode == 'large_normal':
        # mean of each entry is dim(x)
        x[:s] = np.random.normal(n, 1, s)
    
    if permute:
        x = np.random.permutation(x)
    if normalize:
        x /= la.norm(x)
    return x

def random_matrix(m, n, mode='normal'):
    """
    Generate a random matrix of size `m`-by-`n`.
    `mode` can be 'normal', 'bernoulli', 'bad', 'ones', 'hilbert',
    and selects the distribution of the entries.
    """
    if mode == 'bernoulli':
        q = 0.5
        return 1 / np.sqrt(m) * bernoulli.rvs(q, size=(m, n))
    elif mode == 'bad':
        A = 1 / np.sqrt(m) * np.random.normal(0, 1, [m, n])
        U, S, V = np.linalg.svd(A)
        S[1] = 20
        S = np.diag(S)
        return U @ S @ V.T
    elif mode == 'normal':
        return 1 / np.sqrt(m) * np.random.normal(0, 1, [m, n])
    elif mode == 'biased_gaussian':
        return 1 / np.sqrt(m) * np.random.normal(20, 1, [m, n])
    elif mode == 'ones':
        return 100 * np.ones((m, n))
    elif mode == 'hilbert':
        return hilbert(m, n)


def Data_Generate_Positive(x_dim, y_dim, s_sparse, matrix_mode, vector_mode):
    if s_sparse > y_dim:
        raise ValueError(
            f"Invalid sparsity values: {s_sparse = }, {y_dim = }."
            " Please input s_sparse <= y_dim."
        )

    x = random_vector(x_dim, s_sparse, mode=vector_mode, normalize=True, permute=False)
    x = abs(x)

    A = random_matrix(y_dim, x_dim, matrix_mode)

    y = A @ x
    # y = A @ x + np.random.normal(0, .001, y_dim)
    # y = A @ x + np.append(np.array([1, -1]), np.zeros(y_dim-2))
    return A, x, y
