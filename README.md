# NNLS

This package contains some implementations of gradient descent and nesterov acceleration for solving NNLS (Non-negative Least Squares) problems.

Most functions can be found in `nnls.py`.

The input to the functions include a matrix-like `A` and a vector `b`.
All that is required from `A` is that it supports matrix-vector multiplication `A @ x` and conjugation `A.H` returning an object which also supports matrix-vector multiplication.
If `A` is passed as a `numpy ndarray`, it is automatically converted to a `LinOp` object.
As an example, the implementation of a class for subsampled Fourier transform is included in `operators.py`.
