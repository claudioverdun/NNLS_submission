#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sys import path
path.append('../')
from src.operators import LinOp, SubFFT

def build_fft_matrix(n):
    A = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        A[:,i] = np.fft.fft(e)
    return A

def test(x_dim, y_dim):
    idxs = np.random.choice(x_dim, y_dim, replace=False)

    A_FFT = build_fft_matrix(x_dim)
    M_FFT = LinOp(A_FFT[idxs,:]/np.sqrt(y_dim))
    Bop = SubFFT(x_dim, idxs)

    v = np.random.randn(x_dim)
    u1 = (A_FFT @ v)[idxs]/np.sqrt(y_dim)
    u2 = Bop @ v
    u3 = M_FFT @ v
    assert np.allclose(u1, u2)
    assert np.allclose(u1, u3)

    u = np.random.randn(y_dim)
    v1 = A_FFT[idxs,:].conj().T @ u / np.sqrt(y_dim)
    v2 = Bop.H @ u
    v3 = M_FFT.H @ u
    assert np.allclose(v1, v2)
    assert np.allclose(v1, v3)
    print(f"Tests passed for {x_dim=}, {y_dim=}")

test(100, 50)
test(1000, 100)
test(1234, 321)
test(1000, 500)
