#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la

from src.operators import LinOp

########## GD ##########

def gd(A, b, alpha, eta, T,
        L=2, gamma=0, riemannian=True,
        save_x=True, save_e=True):
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

    for t in range(T):
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

    return x, x_record, T, e_record


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

# Direct application of Nesterov acceleration on the original gradient flow
def gd_2layers_nesterov(
        A, b, alpha, eta, T,
        restart=True,
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
        # Gradient for  ||Ax.^2 - b||^2  at  v
        err = A @ (v*v) - b
        if save_x:
            x_record[t] = u*u
        if save_e:
            e_record[t] = la.norm(A @ (u*u) - b)

        z = A.H @ err
        grad = np.multiply(z, v)
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

    return u*u, x_record, T, e_record

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
        err = A @ x - b
        grad = A.H @ err

        zeta -= eta * (k+2)/2 * grad
        
        if save_x:
            x_record[k] = x
        if save_e:
            e_record[k] = la.norm(err)

    return x, x_record, T, e_record

########## (Accelerated) Projected Gradient ##########

def pg(A, b, alpha, eta, T,
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
        x -= eta*z
        x[x < 0] = 0
        
    return x, x_record, T, e_record

def apg(A, b, alpha, eta, T,
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
        p = x + (t/(t+3))*(x_new - x) # Predictor step
        x = x_new
    
    return x, x_record, T, e_record
