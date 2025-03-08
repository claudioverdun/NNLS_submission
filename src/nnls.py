#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import time, tracemalloc

from src.operators import LinOp

########## GD ##########

def gd(A, b, alpha, eta, T,
        L=2, gamma=0, riemannian=True,
        save_x=True, save_e=True,
        tol=1e-6,
        trace_mem=False, trace_time=False, sample_freq=1):
    """ Gradient Descent for minimizing  ||Ax - b||^2  with  x = u^L, starting from  u = alpha*1
    Step-size  eta/(t+2)^gamma  and number of iterations  T
    Riemannian gradient flow if  riemannian=True
    Save  x  (resp.  ||Ax - b||)  if  save_x  (resp.  save_e)  are True
    """

    assert L >= 1
    assert 0 <= gamma <= 1
    q = 2 - 2/L
    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    u  = alpha * np.ones(x_dim)
    x = u**L

    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None
    dt_record = np.zeros(T) if trace_time else None
    mem_record = np.zeros(T) if trace_mem else None
    peak_mem = np.zeros(T) if trace_mem else None

    for t in range(T):
        if t % sample_freq == 0:
            if trace_mem:
                tracemalloc.start()
            if trace_time:
                start_time = time.perf_counter()
        err = (A @ x - b)
        if save_x:
            x_record[t] = x
        if save_e:
            e_record[t] = la.norm(err)

        z = A.H @ err
        if riemannian:
            # Discretization of gradient flow at  x = u^L  (with the \alpha-power metric)
             # Use  L*eta  to match the first order (in eta*u) gradient of  ||Ax.^L - b||^2  at  u  (see below)
            x = x - (L*eta/(t+2)**gamma)*np.multiply(z, x**q)
        else:
            # Discretization of gradient flow at  u  (with the standard Euclidian metric)
            u = u - (eta/(t+2)**gamma)*np.multiply(z, u**(L-1))
            x = u**L
            # = (u - eta * z*u**(L-1) ) ** L
            # = u**L - L*eta * z*u**(2L-2) + O(eta^2 * z**2 * u**(3L-4))
            # = x - L*eta * z * x**q + O(eta^2 * z**2 * x**(2q-1))
            # Cf. Riemannian flow
            # = x - L*eta * z * x**q
        
        if t % sample_freq == 0:
            if trace_mem:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_record[t] = current
                peak_mem[t] = peak
            if trace_time:
                dt_record[t] = time.perf_counter() - start_time
        if la.norm(err) < tol:
            T = t
            break


    return x, x_record, T, e_record, dt_record, mem_record, peak_mem

def gd_bb(A, b, alpha, eta, T,
        L=2, longstep=True, shortstep=False, riemannian=True,
        save_x=True, save_e=True, save_eta=True,
        tol=1e-6,
        trace_mem=False, trace_time=False, sample_freq=1):
    """ Gradient Descent for minimizing  ||Ax - b||^2  with  x = u^L, starting from  u = alpha*1
    Barzilai-Borwein step-size and number of iterations  T
    Riemannian gradient flow if  riemannian=True
    Save  x  (resp.  ||Ax - b||)  if  save_x  (resp.  save_e)  are True
    """

    assert L >= 1
    q = 2 - 2/L
    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    u  = alpha * np.ones(x_dim)
    x = u**L

    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None
    eta_record = np.zeros(T) if save_eta else None
    dt_record = np.zeros(T) if trace_time else None
    mem_record = np.zeros(T) if trace_mem else None
    peak_mem = np.zeros(T) if trace_mem else None

    gprev = np.zeros(x_dim)
    for t in range(T):
        if t % sample_freq == 0:
            if trace_mem:
                tracemalloc.start()
            if trace_time:
                start_time = time.perf_counter()
        err = (A @ x - b)
        if save_x:
            x_record[t] = x
        if save_e:
            e_record[t] = la.norm(err)

        z = A.H @ err
        if riemannian:
            # Discretization of gradient flow at  x = u^L  (with the \alpha-power metric)
            # Multiply by  L  to match the gradient of  ||Au.^L - b||^2  at  u  (see gd)
            grad = L*np.multiply(z, x**q)
            dx = - eta*grad
            x = x + dx
            if t > 1:
                dg = grad - gprev
                if longstep:
                    eta = np.sum(dx*dx) / np.linalg.norm(A @ dx)**2
                elif shortstep:
                    eta = np.linalg.norm(A @ dx)**2 / np.sum(dg*dg)
                eta = max(1e-6, min(eta, 1e6))
            gprev = grad
        else:
            # Discretization of gradient flow at  u  (with the standard Euclidian metric)
            grad = np.multiply(z, u**(L-1))
            du = - eta*grad
            u = u + du
            x = u**L
            if t > 10:
                dg = grad - gprev
                if longstep:
                    eta = np.sum(du*du) / np.linalg.norm(A @ du)**2
                elif shortstep:
                    eta = np.linalg.norm(A @ du)**2 / np.sum(dg*dg)
                # print(np.sum(du*du), np.sum(du*dg), np.sum(dg*dg), ":", eta)
                eta = max(1e-6, min(eta, 1e6))
            gprev = grad
        
        if save_eta:
            eta_record[t] = eta
        if t % sample_freq == 0:
            if trace_mem:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_record[t] = current
                peak_mem[t] = peak
            if trace_time:
                dt_record[t] = time.perf_counter() - start_time
        
        if la.norm(err) < tol:
            T = t
            break

    return x, x_record, T, e_record, eta_record, dt_record, mem_record, peak_mem


########## Nesterov ##########

# Simple discretization of the Riemannian gradient flow
#   y = x + t/2 * x'
#   y' = - t/2 * y * \nabla loss(x)
# with time-step dt = eta
def gd_2layers_nesterov_simple(
        A, b, alpha, eta, T,
        restart=True,
        save_x=True, save_e=True):

    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    u = alpha * np.ones(x_dim)
    u2 = u*u

    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    zeta = u2
    k = 0
    for t in range(T):
        # Riemannian Gradient for  ||Ax - b||^2  at  u2
        err = A @ u2 - b
        if save_x:
            x_record[t] = u2
        if save_e:
            e_record[t] = la.norm(err)

        z = A.H @ err
        grad = np.multiply(z, zeta)
        # Nesterov Acceleration
        zeta = zeta - eta*(k+1)/2*grad
        u2 = u2 + eta*2/(k+1)*(zeta-u2)
        k = k+1
        if restart:
            new_error = la.norm(A@u2 - b)
            if k > 10:
                if new_error > old_error:
                    k = 1
                    zeta = u2
            old_error = new_error

    return u2, x_record, T, e_record

# Direct application of Nesterov acceleration on the reparameterized gradient flow
def gd_nesterov(
        A, b, alpha, eta, T,
        L=2, restart=True, tol=1e-6,
        save_x=True, save_e=True):

    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    u = alpha * np.ones(x_dim)

    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    v = u
    k = 0
    kmin = 10
    for t in range(T):
        # Gradient for  1/4 ||Ax.^L - b||^2  at  v
        err = A @ (v**L) - b
        if save_x:
            x_record[t] = u**L
        if save_e:
            e_record[t] = la.norm(A @ (u**L) - b)

        z = A.H @ err
        grad = (L/2) * np.multiply(z, v)
        # Nesterov Acceleration
        u_new = v - eta * grad
        u_diff = u_new - u
        v = u_new + (k/(k+3))*(u_diff)
        u = u_new
        k = k+1
        if restart:
            if k > kmin:
                if la.norm(u_diff) < la.norm(u_diff_old):
                    k = 1
                    v = u
            u_diff_old = u_diff
        if la.norm(err) < tol:
            T = t
            break

    return u**L, x_record, T, e_record

# Naïve application of Nesterov acceleration to stochastic gradient descent
def sgd_nesterov(
        A, b, alpha, eta, T, L=2,
        save_x=True, save_e=True):

    x_dim = A.shape[1]
    u = alpha * np.ones(x_dim)

    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    v = u
    k = 0
    for t in range(T):
        # Stochastic gradient for  1/4 ||Ax.^L - b||^2  at  v
        i = np.random.randint(A.shape[0])
        a = A[i]
        err = a @ (v**L) - b[i]
        z = a.T * err
        if save_x:
            x_record[t] = u**L
        if save_e:
            e_record[t] = la.norm(A @ (u**L) - b)

        grad = np.multiply(z, v)
        # Nesterov Acceleration
        u_new = v - (L/2) * eta * grad
        u_diff = u_new - u
        v = u_new + (k/(k+3))*(u_diff)
        u = u_new
        k = k+1

    return u**L, x_record, T, e_record

# Discretization of the Riemannian gradient flow
#   y = x + t/2 * x'
#   y' = - t/2 * y * \nabla loss(x)
# after a change of coordinates zeta = log(y)
# with time-step dt = eta
def gd_2layers_nesterov_mirror(
        A, b, alpha, eta, T,
        save_x=True, save_e=True):

    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    x = alpha * np.ones(x_dim)

    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    zeta = np.log(x)
    for k in range(T):
        x = (x + 2/(k+2) * np.exp(zeta))/(1 + 2/(k+2))
        # x = x + 2/(k+1) * (np.exp(zeta) - x)
        err = A @ x - b
        grad = A.H @ err

        zeta -= eta * (k+2)/2 * grad
        # zeta -= eta * (k+1)/2 * grad
        
        if save_x:
            x_record[k] = x
        if save_e:
            e_record[k] = la.norm(err)

    return x, x_record, T, e_record

########## (Accelerated) Projected Gradient ##########

def pg(A, b, alpha, eta, T,
       tol=1e-6,
       save_x=True, save_e=True):
    """ Projected Gradient for minimizing  ||Ax - b||^2 , starting from  x = alpha*1
    Step-size  eta  and number of iterations  T
    Save  x  (resp.  ||Ax - b||)  if  save_x  (resp  save_e)  are True
    """
    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    x  = alpha * np.ones(x_dim)
    
    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    for t in range(T):
        err = (A @ x - b)
        if save_x:
            x_record[t] = x
        if save_e:
            e_record[t] = la.norm(err)

        z = A.H @ err
        x -= eta*z.real
        x[x < 0] = 0
        if la.norm(err) < tol:
            T = t
            break
        
    return x, x_record, T, e_record

def apg(A, b, alpha, eta, T,
        tol=1e-6,
        save_x=True, save_e=True):
    """ Accelerated Projected Gradient for minimizing  ||Ax - b||^2 , starting from  x = alpha*1
    Step-size  eta  and number of iterations  T
    Save  x  (resp.  ||Ax - b||)  if  save_x  (resp  save_e)  are True

    Cf Polyak 2015, with notation p for the predictor x_k, x for the corrector X_k
    """
    if isinstance(A, np.ndarray):
        A = LinOp(A)
    x_dim = A.shape[1]
    x  = alpha * np.ones(x_dim)
    p = np.copy(x)
    
    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    for t in range(T):
        err = (A @ p - b)
        if save_x:
            x_record[t] = x
        if save_e:
            e_record[t] = la.norm(err)

        grad = A.H @ err
        x_new = p - eta*grad # Corrector step
        x_new[x_new < 0] = 0 # Projection step on positive orthant
        p = x_new + (t/(t+3))*(x_new - x) # Predictor step
        x = x_new
        if la.norm(err) < tol:
            T = t
            break
    
    return x, x_record, T, e_record

### Accelerated SGD from https://arxiv.org/pdf/1803.05591
def asgd(A, b, alpha, eta, T,
         L=2, ξ=1, κ=1,
        save_x=True, save_e=True):
    """ Accelerated Stochastic Gradient Descent for minimizing  ||Av**L - b||^2 , starting from  x = alpha*1
    Step-size  eta  and number of iterations  T
    Long-step parameter  κ \geq 1  and statistical advantage parameter  ξ \leq \sqrt{κ}
    Save  x = v**L  (resp.  ||Av**L - b||)  if  save_x  (resp  save_e)  are True
    """
    x_dim = A.shape[1]
    v = alpha * np.ones(x_dim)
    w = np.copy(v)
    
    x = v**L
    x_record = np.zeros([T, x_dim]) if save_x else None
    e_record = np.zeros(T) if save_e else None

    α = 1 - 0.7**2 * ξ / κ
    δ = eta

    for t in range(T):
        i = np.random.randint(A.shape[0])
        a = A[i]
        err = a @ (v**L) - b[i]
        z = a.T * err
        grad = np.multiply(z, v**(L-1))

        w = α*w + (1-α)*(v - κ*δ/0.7 * grad)
        v = 0.7/(0.7 + (1-α)) * (v - δ*grad) + (1-α)/(0.7 + (1-α)) * w

        if save_x:
            x_record[t] = v**L
        if save_e:
            e_record[t] = la.norm(A @ (v**L) - b)

    return x, x_record, T, e_record
