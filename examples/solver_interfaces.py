import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time

from scipy.optimize import nnls as lh
import cvxpy as cp

from sys import path
path.append('../')

from src import nnls
from pytntnn.tntnn import tntnn

import datagen

# Reviewer comment
# Comparision with CVX-NNLS, LH-NNLS, TNT-NN. To compare this method with classic NNLS solver, the authors should also include the training time as well. Currently, it is reported in the paper that this overparametrization method converges with better accuracy, but is is unclear how fast it is compared with other solvers.

# Our promise
# We are currently conducting a comprehensive series of timing experiments across multiple problem scales and scenarios. Our experimental framework includes:

# Small-scale problems (N < 500): Testing against optimized implementations of LH-NNLS and TNT-NN with varying matrix densities and condition numbers.
# Medium-scale problems (500 < N < 2000): Comparing performance across different types of data distributions and sparsity patterns.
# Large-scale problems (N > 2000): Evaluating scalability and computational efficiency, particularly for problems where traditional methods face memory constraints.

# For each problem size, we are running multiple trials to capture performance variability, and our results will include error bars showing the standard deviation in computation time. In any case, it is important to point out that traditional solvers like LH-NNLS and TNT-NN perform faster per iteration since they are a wrapper of optimized C++/FORTRAN implementations, while our method is implemented in pure Python and it is not optimized with low-level languages. However, those algorithms don't have any theoretical guarantees besides convergence on a finite number of steps.

# Preliminary findings indicate that while traditional solvers have an advantage in small problems due to their optimized implementations, our method becomes increasingly competitive as the problem size grows, primarily because (i.) our algorithm requires only matrix-vector operations rather than solving linear systems, (ii.) the memory footprint remains manageable even for large problems, (iii.) the operations are naturally parallelizable, (iv.) we avoid the computational overhead of active set management.

# The complete experimental results, including detailed timing analyses with variance measurements and scaling behavior across different problem dimensions, will be included in the revised manuscript. Still, our preliminary experiments show that BB step sizes provide significant speedups over constant step sizes. In particular, a key finding from our experiments is that GD-2L and GD-3L require significantly fewer iterations than PGD to achieve the same accuracy - for instance, our methods typically need O(100) iterations to reach an error of 1e-3, while PGD requires O(1000) iterations. This order-of-magnitude improvement in iteration count is a crucial practical advantage.

#
# Simple algorithms: take only A, b
#
simple = ['CvxBP', 'CvxNNLS', 'LH', 'TNT']
def solve_lh(A, b):
    start = time.time()
    x, err = lh(A, b, atol=1e-6)
    end = time.time()
    return x, err, end-start

def solve_tnt(A, b):
    start = time.time()
    tnt_result = tntnn(A, b, rel_tol=1e-6)
    end = time.time()
    x = tnt_result.x
    return x, la.norm(A@x - b), end-start

def solve_bp(A, b):
    x_dim = A.shape[1]
    x = cp.Variable(x_dim)
    objective = cp.Minimize(cp.norm(x,1))
    constraints = [A@x == b]
    prob = cp.Problem(objective, constraints)
    start = time.time()
    result = prob.solve()
    end = time.time()
    return x.value, la.norm(A@(x.value) - b), end-start

def solve_cvxnnls(A, b):
    x_dim = A.shape[1]
    z = cp.Variable(x_dim)
    objective = cp.Minimize(cp.sum_squares(A@z-b))
    constraints = [z >= 0]
    prob = cp.Problem(objective, constraints)
    start = time.time()
    result = prob.solve()
    end = time.time()
    return z.value, la.norm(A@(z.value) - b), end-start

#
# PG and APG: take A, b; then initial point, stepsize, max_iter
#
pg_solvers = ['PG', 'APG']
def solve_pg(A, b, alpha, eta, T):
    start = time.time()
    x, _, T, *_ = nnls.pg(A, b, alpha, eta, T, save_x=False, save_e=False)
    end = time.time()
    return x, la.norm(A@x - b), end-start, T

def solve_apg(A, b, alpha, eta, T):
    start = time.time()
    x, _, T, *_ = nnls.apg(A, b, alpha, eta, T, save_x=False, save_e=False)
    end = time.time()
    return x, la.norm(A@x - b), end-start, T

#
# Reparametrized NNLS: take A, b; then initial point, stepsize, max_iter;
# then number of layers, and riemannian flag
#
ours = ['GD', 'GD-BB', 'GD-Nesterov', 'MD-Nesterov']
def solve_gd(A, b, alpha, eta, T, L, riemannian=False):
    start = time.time()
    x, _, T, *_ = nnls.gd(A, b, alpha**(1/L), eta, T, L=L, save_x=False, save_e=False, riemannian=riemannian)
    end = time.time()
    return x, la.norm(A@x - b), end-start, T

def solve_gd_bb(A, b, alpha, eta, T, L, riemannian=False):
    start = time.time()
    x, _, T, *_ = nnls.gd_bb(A, b, alpha**(1/L), eta, T, L=L, save_x=False, save_e=False, riemannian=riemannian)
    end = time.time()
    return x, la.norm(A@x - b), end-start, T

def solve_gd_nesterov(A, b, alpha, eta, T, L, riemannian=False):
    start = time.time()
    x, _, T, *_ = nnls.gd_nesterov(A, b, alpha**(1/L), eta, T, L=L, save_x=False, save_e=False)
    end = time.time()
    return x, la.norm(A@x - b), end-start, T

def solve_mirror_nesterov(A, b, alpha, eta, T, L, riemannian=False):
    start = time.time()
    x, _, T, *_ = nnls.gd_2layers_nesterov_mirror(A, b, alpha**(1/2), eta, T, save_x=False, save_e=False)
    end = time.time()
    return x, la.norm(A@x - b), end-start, T

solvers = {
    'CvxBP': solve_bp, 'CvxNNLS': solve_cvxnnls,
    'LH': solve_lh, 'TNT': solve_tnt,
    'PG': solve_pg, 'APG': solve_apg,
    'GD': solve_gd, 'GD-BB': solve_gd_bb,
    'GD-Nesterov': solve_gd_nesterov, 'MD-Nesterov': solve_mirror_nesterov
    }

sparse_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
dense_alphas = [1e-2, 1e-1, 1]
Ls = [2, 3]
def basic_test(x_dim, y_dim, s, seed,
               eta=0.1, T=int(1e4), alphas=None,
               nreps=10, verbose=0):
    np.random.seed(seed)

    results = {}
    results['x_dim'] = x_dim
    results['y_dim'] = y_dim
    results['s'] = s
    results['seed'] = seed
    results['Targets'] = []

    if alphas is None:
        if y_dim < x_dim:
            alphas = sparse_alphas
        else:
            alphas = dense_alphas

    # Initialize results
    for alg in simple:
        results[alg] = []
    for alg in pg_solvers:
        for alpha in alphas:
            results[(alg, alpha)] = []
    for alg in ours:
        for L in Ls:
            for alpha in alphas:
                results[(alg, L, alpha)] = []

    for k in range(nreps):
        print(k)
        A, x_ground, b = datagen.Data_Generate_Positive(x_dim, y_dim, s, 'normal', 'normal')

        results['Targets'].append([x_ground, la.norm(A@x_ground - b), 0])
        for solver in simple:
            x, err, dt = solvers[solver](A, b)
            results[solver].append([x, err, dt, 1])
            if verbose > 0:
                print(f"{solver:>8} {err=:.3e}, {dt=:.5f}", end="")
                if verbose > 1:
                    print(f" L1: {la.norm(x,1):.4f}")
                else:
                    print()

        for solver in pg_solvers:
            for alpha in alphas:
                x, err, dt, n = solvers[solver](A, b, alpha, eta, T)
                results[(solver, alpha)].append([x, err, dt, n])
                if verbose > 0:
                    print(f"{solver:>8} {alpha=:.3e}: {err=:.3e}, {dt=:.5f}", end="")
                    if verbose > 1:
                        print(f" L1: {la.norm(x,1):.4f}, {n=:4}")
                    else:
                        print()

        for solver in ours:
            for L in Ls:
                for alpha in alphas:
                    x, err, dt, n = solvers[solver](A, b, alpha, eta, T, L, riemannian=True)
                    results[(solver, L, alpha)].append([x, err, dt, n])
                    if verbose > 0:
                        print(f"{solver:>8} {L=} {alpha=:.3e}: {err=:.3e}, {dt=:.5f}", end="")
                        if verbose > 1:
                            print(f" L1: {la.norm(x,1):.4f}, {n=:4}")
                        else:
                            print()
    
    return results

def square_dim_scaling(x_dims, seed,
               eta=0.1, T=int(1e4), alpha=1e-2,
               nreps=10, verbose=0):
    results = {}
    results['x_dims'] = x_dims
    results['seed'] = seed

    square_solvers = ['LH', 'TNT', 'PG', 'APG', 'GD', 'GD-BB', 'GD-Nesterov', 'MD-Nesterov']
    for alg in square_solvers:
        results[alg] = {}

    for x_dim in x_dims:
        print(x_dim)
        for alg in square_solvers:
            results[alg][x_dim] = []
        for k in range(nreps):
            print(k)
            A, x_ground, b = datagen.Data_Generate_Positive(x_dim, x_dim, x_dim, 'normal', 'normal')

            for alg in ['LH', 'TNT']:
                x, err, dt = solvers[alg](A, b)
                results[alg][x_dim].append([x, err, dt, 1])
                if verbose > 0:
                    print(f"{alg:>8} {err=:.3e}, {dt=:.5f}")

            for alg in ['PG', 'APG']:
                x, err, dt, n = solvers[alg](A, b, alpha, eta, T)
                results[alg][x_dim].append([x, err, dt, n])
                if verbose > 0:
                    print(f"{alg:>8} {err=:.3e}, {dt=:.5f}, {n=:4}")

            for alg in ['GD', 'GD-BB', 'GD-Nesterov', 'MD-Nesterov']:
                x, err, dt, n = solvers[alg](A, b, alpha, eta, T, L=2)
                results[alg][x_dim].append([x, err, dt, n])
                if verbose > 0:
                    print(f"{alg:>8} {err=:.3e}, {dt=:.5f}, {n=:4}")

    
    return results


