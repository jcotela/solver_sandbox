import logging

import numpy as np

from scipy import sparse
from scipy.linalg import norm
from scipy.sparse.linalg import cg, spsolve_triangular

from .performance import profile, solver_report_callback as report
from .utilities import check_solution

log = logging.getLogger(__name__)

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
    def __init__(self, A, b, relaxation_w=1.0):
        U = sparse.triu(A, k=1, format='csr')
        L = sparse.tril(A, k=-1, format='csr')
        D = sparse.diags(A.diagonal())
        self.w = relaxation_w
        self.b = self.w * b
        self.L = self.w * L +  D
        self.U = self.w * U + (self.w - 1) * D

    def __call__(self, x):
        return spsolve_triangular(self.L, self.b - self.U @ x)


class Jacobi(object):
    def __init__(self, A, b, relaxation_w=1.0):
        U = sparse.triu(A, k=1, format='csr')
        L = sparse.tril(A, k=-1, format='csr')
        self.LU = L + U
        self.D = A.diagonal()
        self.b = b
        self.w = relaxation_w

    def __call__(self, x):
        y = (1.0 - self.w) * x
        return y + self.w * np.divide(self.b - self.LU @ x, self.D)


def test_gauss_seidel(w=1.0):
    #A = sparse.csr_array(np.array([[16, 3], [7, -11]], dtype=np.float64))
    #b = np.array([11, 13], dtype=np.float64)
    #x = np.array([1.0, 1.0])

    A = sparse.csr_array([
        [10.0, -1.0, 2.0, 0.0],
        [-1.0, 11.0, -1.0, 3.0],
        [2.0, -1.0, 10.0, -1.0],
        [0.0, 3.0, -1.0, 8.0],
    ])
    b = np.array([6.0, 25.0, -11.0, 15.0])
    x = np.array([0.0, 0.0, 0.0, 0.0])

    gs = GaussSeidel(A, b, w)
    for _ in range(10):
        x = gs(x)
        print(x)


def test_jacobi(w=1.0):
    A = sparse.csr_array(np.array([[2, 1], [5, 7]], dtype=np.float64))
    b = np.array([11, 13], dtype=np.float64)
    x = np.array([1.0, 1.0])

    #A = sparse.csr_array([
    #    [10.0, -1.0, 2.0, 0.0],
    #    [-1.0, 11.0, -1.0, 3.0],
    #    [2.0, -1.0, 10.0, -1.0],
    #    [0.0, 3.0, -1.0, 8.0],
    #])
    #b = np.array([6.0, 25.0, -11.0, 15.0])
    #x = np.array([0.0, 0.0, 0.0, 0.0])

    j = Jacobi(A, b, w)
    for _ in range(25):
        x = j(x)
        print(x)

    print(A@x)


def solve_iteration(A, x, residual, inter, restr):

    smoothing_iter = 20
    delta = np.zeros_like(x)

    with profile('coarse matrix calculation'):
        coarse_A = restr @ A @ inter

    #smooth = Jacobi(A, residual, relaxation_w=0.5)
    smooth = GaussSeidel(A, residual, relaxation_w=1.0)

    with profile('smoothing'):
        for _ in range(smoothing_iter):
            delta = smooth(delta)

    with profile('coarse solve'):
        coarse_res = restr @ (residual - A @ delta)
        coarse_delta, _ = cg(coarse_A, coarse_res, atol=1e-6)


    with profile('update solution'):
        delta += inter @ coarse_delta
        x += delta
        residual -= A @ delta

    return x, residual


def solve(A, b, inter, restr):
    residual = np.array(b)
    x = np.zeros_like(b)
    norm_res = norm_b = norm(b)
    tol = 1e-5

    while (norm_res > tol * norm_b):
        x, residual = solve_iteration(A, x, residual, inter, restr)
        norm_res = norm(residual)
        log.info(norm_res)

    return x

if __name__ == '__main__':
    #test_gauss_seidel(1.5)
    test_jacobi()
    exit()
    from pathlib import Path
    from utilities import read_mm

    logging.basicConfig(level=logging.DEBUG)

    with profile('problem load'):
        base_path = Path('./examples/bracket')

        A = read_mm(base_path / 'first_fine_A.mm')
        b = read_mm(base_path / 'first_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')

    x = solve(A, b, inter, restr)

    with profile('solution check'):
        check_solution(A, x, b)

    '''
    GS 1.5
    real    4m46.455s
    user    5m41.484s
    sys     3m7.611s
    GS 1.0
    real    4m47.758s
    user    5m49.839s
    sys     3m37.062s
    '''