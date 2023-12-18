import logging

import numpy as np

from scipy import sparse
from scipy.linalg import norm
from scipy.sparse.linalg import cg, spsolve_triangular

from .performance import profile
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


class BlockSystem(object):
    def __init__(self, A, b, inter, restr, q_inter, q_restr):
        self.inter = inter.trunc()
        self.q_inter = q_inter.trunc()
        self.restr = self.inter.transpose()
        self.q_restr = self.q_inter.transpose()
        self.Q = inter @ self.restr + q_inter @ self.q_restr
        self.invQ = 2 * self.Q.trunc() - self.Q

        self.A = self.Q @ A @ self.invQ

        self.All = self.restr @ A @ self.inter
        self.Aqq = self.q_restr @ A @ self.q_inter
        self.Alq = self.restr @ A @ self.q_inter
        self.Aql = self.Alq.transpose()
        self.res_l = self.restr @ b
        self.res_q = self.q_restr @ b


def solve_iter_separate(bs: BlockSystem, x):

    smoothing_iter = 5
    relax = 0.5

    delta_l = np.zeros_like(bs.res_l)
    delta_q = np.zeros_like(bs.res_q)

    with profile('smoothing'):
        for _ in range(smoothing_iter):
            l_smoother = GaussSeidel(bs.All, bs.res_l - bs.Alq @ delta_q, relax)
            delta_l = l_smoother(delta_l)
            q_smoother = GaussSeidel(bs.Aqq, bs.res_q - bs.Aql @ delta_l, relax)
            delta_q = q_smoother(delta_q)

    with profile('coarse solve'):
        coarse_res = bs.res_l - bs.All @ delta_l - bs.Alq @ delta_q
        coarse_delta, _ = cg(bs.All, coarse_res, atol=1e-1)

    with profile('update solution'):
        delta_l += coarse_delta
        bs.res_l -= bs.All @ delta_l + bs.Alq @ delta_q
        bs.res_q -= bs.Aql @ delta_l + bs.Aqq @ delta_q
        x += bs.inter @ delta_l + bs.q_inter @ delta_q
        residual = bs.inter @ bs.res_l + bs.q_inter @ bs.res_q

    return x, residual

def solve_separate(A, b, inter, restr, q_inter, q_restr):

    with profile('scale separation'):
        bs = BlockSystem(A, b, inter, restr, q_inter, q_restr)

    x = np.zeros_like(b)
    norm_res = norm_b = norm(b)
    tol = 1e-5
    it = 0
    max_iter = 100

    while (norm_res > tol * norm_b and it < max_iter):
        x, residual = solve_iter_separate(bs, x)
        norm_res = norm(residual)
        it += 1
        log.info("%d %f", it, norm_res)
        #log.debug(norm(b - A@x))
        #log.debug(norm(residual))

    return x


if __name__ == '__main__':
    from pathlib import Path
    from .utilities import read_mm

    logging.basicConfig(level=logging.DEBUG)

    with profile('problem load'):
        base_path = Path('./examples/bracket')

        A = read_mm(base_path / 'last_fine_A.mm')
        b = read_mm(base_path / 'last_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')
        q_inter = read_mm(base_path / 'quad_interpolation.mm')
        q_restr = read_mm(base_path / 'quad_restriction.mm')

    x = solve_separate(A, b, inter, restr, q_inter, q_restr)

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