import logging

import numpy as np

from scipy import sparse
from scipy.linalg import norm
from scipy.sparse.linalg import cg, spsolve_triangular

from performance import profile, solver_report_callback as report
from utilities import check_solution

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
        self.U = sparse.triu(A, k=1, format='csr')
        self.L = sparse.tril(A, k=0, format='csr')
        self.b = b
        self.w = relaxation_w

    def __call__(self, x):
        return spsolve_triangular(self.L, self.b - self.U @ x)


def test_gauss_seidel():
    A = sparse.csr_array(np.array([[16, 3], [7, -11]], dtype=np.float64))
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

    gs = GaussSeidel(A, b)
    for _ in range(10):
        x = gs(x)
        print(x)


def solve_iteration(A, x, residual, inter, restr):

    smoothing_iter = 20
    delta = np.zeros_like(x)

    with profile('coarse matrix calculation'):
        coarse_A = restr @ A @ inter

    gs = GaussSeidel(A, residual)

    with profile('smoothing'):
        for _ in range(smoothing_iter):
            delta = gs(delta)

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
    #test_gauss_seidel()
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