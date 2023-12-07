import logging

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg, factorized, LinearOperator

from performance import profile, solver_report_callback as report
from utilities import check_solution

log = logging.getLogger(__name__)

def solve(A, b, inter, restr):

    with profile('coarse factorization'):
        coarse_A = csc_matrix(restr @ A @ inter)
        coarse_solve = factorized(coarse_A)

    with profile('set up deflated problem'):
        Az = A@inter
        def P(x):
            return x - Az @ coarse_solve(restr @ x)

        PA = LinearOperator(A.shape, matvec=lambda x: P(A@x) )

        Pb = P(b)

    with profile('set up preconditioner'):
        diag = A.diagonal()
        precond = LinearOperator(A.shape, lambda x: np.divide(x, diag))

    with profile('deflated CG solve'):

        tolerance = 1e-5
        y, info = cg(PA, Pb, tol=tolerance,
                     callback=report(log_level=logging.DEBUG), M=precond)
        if info != 0:
            log.warning(f'deflated CG solver did not converge! info: {info}')

    with profile('finalize solution'):
        ry = b - A@y
        corr = inter @ coarse_solve(restr @ ry)
        x = y + corr


    return x


if __name__ == '__main__':
    from pathlib import Path
    from utilities import read_mm

    logging.basicConfig(level=logging.DEBUG)

    with profile('problem load'):
        base_path = Path('./examples/bracket')

        A = read_mm(base_path / 'last_fine_A.mm')
        b = read_mm(base_path / 'last_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')

    x = solve(A, b, inter, restr)

    with profile('solution check'):
        check_solution(A, x, b)