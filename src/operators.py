#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

########## Abstract Linear Algebra classes ##########

class LinOp:
    def __init__(self, A):
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.A = A
    
    def __matmul__(self, x):
        return self.A @ x
    
    @property
    def H(self):
        return LinOp(self.A.conj().T)

    @property
    def shape(self):
        return (self.m, self.n)

class SubFFT:
    def __init__(self, dim, idxs):
        self.m = len(idxs)
        self.n = dim
        self.idxs = idxs

    def __matmul__(self, x):
        return np.fft.fft(x)[self.idxs] / np.sqrt(self.m)

    def __rmatmul__(self, x):
        x_lift = np.zeros(self.n, dtype=x.dtype)
        x_lift[self.idxs] = x / np.sqrt(self.m)
        return self.n * np.fft.ifft(x_lift)
    
    @property
    def H(self):
        return SubFFT_H(self.n, self.idxs)

    @property
    def shape(self):
        return (self.m, self.n)

class SubFFT_H:
    def __init__(self, dim, idxs):
        self.m = dim
        self.n = len(idxs)
        self.idxs = idxs

    def __matmul__(self, x):
        x_lift = np.zeros(self.m, dtype=x.dtype)
        x_lift[self.idxs] = x / np.sqrt(self.n)
        return self.m * np.fft.ifft(x_lift)

    def __rmatmul__(self, x):
        return np.fft.fft(x)[self.idxs] / np.sqrt(self.n)

    @property
    def H(self):
        return SubFFT(self.m, self.idxs)
    
    @property
    def shape(self):
        return (self.m, self.n)
