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
        int_ = inter.trunc()

        #self.invQ = int_ @ restr #+ q_inter @ q_restr
        self.invQ = inter @ int_.transpose() + q_inter @ q_restr
        log.debug(inter.nnz)
        log.debug((self.invQ @ int_ != inter).nnz)
        assert((self.invQ @ int_ != inter).nnz == 0)
        #self.Q = 2 * self.invQ.trunc() - self.invQ
        self.Q = self.invQ.transpose()

        self.inter = inter
        self.restr = int_.transpose() @ self.Q

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
        # Some sanity checks (efficient comparison of equality for CSR)
        assert((restr != inter.transpose()).nnz == 0)
        assert((q_restr != q_inter.transpose()).nnz == 0)

        self.invQ = inter @ self.restr + q_inter @ q_restr
        self.invQt = self.invQ.transpose()
        self.Q = 2 * self.invQ.trunc() - self.invQ
        from scipy import sparse
        idt = sparse.eye(*self.Q.shape)
        assert(((self.Q @ self.invQ) != idt).nnz == 0)

        self.A = self.invQt @ A @ self.invQ
        self.residual = self.invQt @ b
        self.b = self.invQt @ b

        self.All = self.restr @ self.A @ self.inter
        #self.res_l = self.restr @ self.residual

    def postprocess_solution(self, x):
        return self.invQ @ x

def solve_iteration(bs, smooth, x, cg_tolerance):

    smoothing_iter = 5
    delta = np.zeros_like(x)

    with profile('smoothing'):
        for _ in range(smoothing_iter):
            delta = smooth(delta, bs.residual)

    with profile('coarse solve'):
        bs.res_l = bs.restr @ (bs.residual - bs.A @ delta)
        coarse_delta, i = cg(bs.All, bs.res_l, atol=cg_tolerance, maxiter=500)

    if i:
        log.warning("CG did not converge (%d)" % i)

    with profile('update solution'):
        delta += bs.inter @ coarse_delta
        x += delta
        bs.residual -= bs.A @ delta
        log.debug(norm(bs.residual))
        #log.debug(norm(bs.b - bs.A @ x))

    return x


def solve(bs, max_iter = 10, tolerance = 1e-5, cg_tolerance = 1e-5):
    x = np.zeros_like(b)
    it = 0

    smooth = GaussSeidel(bs.A, relaxation_w=0.5)

    norm_res = norm_b = norm(bs.residual)

    while (norm_res > tolerance * norm_b and it < max_iter):
        x = solve_iteration(bs, smooth, x, cg_tolerance)
        norm_res = norm(bs.residual)
        it += 1
        log.info("%d %f", it, norm_res)
        #log.debug(norm(b - A@bs.postprocess_solution(x)))
        #log.debug("%f %f", norm(bs.residual), 0.0)#norm(bs.invQ @ bs.residual))
        #log.debug("%f %f", norm(bs.residual), norm(bs.invQ @ bs.residual))


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

    logging.basicConfig(level=logging.WARNING)

    with profile('problem load'):
        base_path = Path('./examples/bracket')

        A = read_mm(base_path / 'first_fine_A.mm')
        b = read_mm(base_path / 'first_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')
        q_inter = read_mm(base_path / 'quad_interpolation.mm')
        q_restr = read_mm(base_path / 'quad_restriction.mm')

    problems = [
        ("raw problem", RawSystem(A, b, inter, restr, q_inter, q_restr)),
        ("mixed problem", MixedSystem(A, b, inter, restr, q_inter, q_restr)),
        (
            "pure hierarhical",
            PureHierarchicalSystem(A, b, inter, restr, q_inter, q_restr)
        )
    ]

    for label, system in problems:
        with profile(label, log_level=logging.WARNING):
            x = solve(system, max_iter=50, cg_tolerance=1e-1)

        with profile('solution check'):
            check_solution(A, x, b, log_level=logging.WARNING)
