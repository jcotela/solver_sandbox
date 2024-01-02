import logging

import numpy as np

from scipy.linalg import norm
from scipy.sparse.linalg import cg

from .performance import profile
from .utilities import check_solution
from .smoothing import GaussSeidel, Jacobi

log = logging.getLogger(__name__)


class RawSystem(object):
    """
    Lagrange coarse + Hierarchical fine.
    Raw matrices read from C++
    """
    def __init__(self, A, b, inter, restr, q_inter, q_restr):
        self.A = A
        self.residual = b.copy()

        self.All = restr @ A @ inter
        self.res_l = restr @ b

        self.inter = inter
        self.restr = restr

    def postprocess_solution(self, x):
        return x


class MixedSystem(object):
    """
    Lagrange coarse + Hierarchical fine.
    Recalculated matrices
    """
    def __init__(self, A, b, inter, restr, q_inter, q_restr):
        int = inter.trunc()

        self.Q = int @ restr + q_inter @ q_restr
        self.invQ = 2 * self.Q.trunc() - self.Q

        self.inter = inter
        self.restr = int.transpose() @ self.Q

        self.A = A
        self.residual = b.copy()

        self.All = self.restr @ A @ self.inter
        self.res_l = self.restr @ b

    def postprocess_solution(self, x):
        return x


class PureHierarchicalSystem(object):
    """
    Hierarchical coarse + Hierarchical fine.
    Recalculated matrices
    """
    def __init__(self, A, b, inter, restr, q_inter, q_restr):
        self.inter = inter.trunc()
        self.restr = self.inter.transpose()

        self.Q = inter @ self.restr + q_inter @ q_restr
        self.invQ = 2 * self.Q.trunc() - self.Q

        self.A = self.Q @ A @ self.invQ
        self.residual = self.Q @ b

        self.All = self.restr @ self.A @ self.inter
        self.res_l = self.restr @ self.residual

    def postprocess_solution(self, x):
        return self.invQ @ x

def solve_iteration(bs, smooth, x):

    smoothing_iter = 5
    delta = np.zeros_like(x)

    with profile('smoothing'):
        for _ in range(smoothing_iter):
            delta = smooth(delta, bs.residual)

    with profile('coarse solve'):
        bs.res_l = bs.restr @ (bs.residual - bs.A @ delta)
        coarse_delta, _ = cg(bs.All, bs.res_l, atol=1e-5)


    with profile('update solution'):
        delta += bs.inter @ coarse_delta
        x += delta
        bs.residual -= bs.A @ delta

    return x


def solve(bs):
    x = np.zeros_like(b)
    tol = 1e-5
    it = 0
    max_iter = 10

    smooth = GaussSeidel(bs.A, relaxation_w=0.5)

    norm_res = norm_b = norm(bs.residual)

    while (norm_res > tol * norm_b and it < max_iter):
        x = solve_iteration(bs, smooth, x)
        norm_res = norm(bs.residual)
        it += 1
        log.info("%d %f", it, norm_res)
        log.debug(norm(b - A@bs.postprocess_solution(x)))
        log.debug("%f %f", norm(bs.residual), 0.0)#norm(bs.invQ @ bs.residual))

    return bs.postprocess_solution(x)


class CoarseSystem(object):
    def __init__(self, A, b, inter, restr, q_inter, q_restr, relaxation_w=1.0):
        int = inter.trunc()

        self.Q = int @ restr + q_inter @ q_restr
        self.invQ = 2 * self.Q.trunc() - self.Q

        self.inter = inter
        self.restr = int.transpose() @ self.Q

        self.A = A
        self.All = self.restr @ A @ self.inter
        self.res_l = self.restr @ b


class BlockSystem(object):
    def __init__(self, A, b, inter, restr, q_inter, q_restr, relaxation_w=1.0):
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

        self.l_smoother = GaussSeidel(self.All, relaxation_w)
        self.q_smoother = GaussSeidel(self.Aqq, relaxation_w)


def solve_iter_separate(bs: BlockSystem, x):

    smoothing_iter = 5

    delta_l = np.zeros_like(bs.res_l)
    delta_q = np.zeros_like(bs.res_q)

    with profile('smoothing'):
        for _ in range(smoothing_iter):
            delta_l = bs.l_smoother(delta_l, bs.res_l - bs.Alq @ delta_q)
            delta_q = bs.q_smoother(delta_q, bs.res_q - bs.Aql @ delta_l)

    with profile('coarse solve'):
        coarse_res = bs.res_l - bs.All @ delta_l - bs.Alq @ delta_q
        coarse_delta, _ = cg(bs.All, coarse_res, atol=1e-5)

    with profile('update solution'):
        delta_l += coarse_delta
        bs.res_l -= bs.All @ delta_l + bs.Alq @ delta_q
        bs.res_q -= bs.Aql @ delta_l + bs.Aqq @ delta_q
        x += bs.inter @ delta_l + bs.q_inter @ delta_q
        residual = bs.inter @ bs.res_l + bs.q_inter @ bs.res_q

    return x, residual

def solve_separate(A, b, inter, restr, q_inter, q_restr):

    with profile('scale separation'):
        relax = 0.5
        bs = BlockSystem(A, b, inter, restr, q_inter, q_restr, relax)

    x = np.zeros_like(b)
    norm_res = norm_b = norm(b)
    tol = 1e-5
    it = 0
    max_iter = 10

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

        A = read_mm(base_path / 'first_fine_A.mm')
        b = read_mm(base_path / 'first_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')
        q_inter = read_mm(base_path / 'quad_interpolation.mm')
        q_restr = read_mm(base_path / 'quad_restriction.mm')

    bs = RawSystem(A, b, inter, restr, q_inter, q_restr)
    #bs = PureHierarchicalSystem(A, b, inter, restr, q_inter, q_restr)
    #bs = BlockSystem(A, b, inter, restr, q_inter, q_restr)
    #x = solve_separate(A, b, inter, restr, q_inter, q_restr)
    x = solve(bs)

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