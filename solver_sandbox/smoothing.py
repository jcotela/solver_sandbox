import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular


class GaussSeidel_ew(object):
    def __init__(self, A, b, relaxation_w=1.0):
        self.A = A
        self.b = b
        self.w = relaxation_w

    def __call__(self, x):
        indptr = self.A.indptr
        indices = self.A.indices
        data = self.A.data
        diag = 1.0
        b = self.b

        for row in range(self.A.shape[0]):
            y = b[row]
            for col,value in zip(
                    indices[indptr[row]:indptr[row+1]],
                    data[indptr[row]:indptr[row+1]]):
                if row == col:
                    diag = value
                else:
                    y -= value * x[col]
            x[row] = y / diag

        return x


class GaussSeidel(object):
    def __init__(self, A, relaxation_w=1.0):
        U = sparse.triu(A, k=1, format='csr')
        L = sparse.tril(A, k=-1, format='csr')
        D = sparse.diags(A.diagonal())
        self.w = relaxation_w
        self.L = self.w * L +  D
        self.U = self.w * U + (self.w - 1) * D

    def __call__(self, x, b):
        return spsolve_triangular(self.L, self.w * b - self.U @ x)


class Jacobi(object):
    def __init__(self, A, relaxation_w=1.0):
        U = sparse.triu(A, k=1, format='csr')
        L = sparse.tril(A, k=-1, format='csr')
        self.LU = L + U
        self.D = A.diagonal()
        self.w = relaxation_w

    def __call__(self, x, b):
        y = (1.0 - self.w) * x
        return y + self.w * np.divide(b - self.LU @ x, self.D)