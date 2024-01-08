import logging

import numpy as np

from scipy.sparse.linalg import cg, LinearOperator

from .performance import profile, solver_report_callback as report
from .utilities import check_solution

log = logging.getLogger(__name__)

def solve(A, b):

    with profile('set up preconditioner'):
        diag = A.diagonal()
        precond = LinearOperator(A.shape, lambda x: np.divide(x, diag))

    with profile('deflated CG solve'):

        tolerance = 1e-5
        x, info = cg(A, b, tol=tolerance,
                     callback=report(log_level=logging.DEBUG), M=precond)
        if info != 0:
            log.warning(f'preconditioned CG solver did not converge! info: {info}')

    return x


if __name__ == '__main__':
    from pathlib import Path
    from .utilities import read_mm

    logging.basicConfig(level=logging.DEBUG)

    with profile('problem load'):
        base_path = Path('./examples/bracket')

        A = read_mm(base_path / 'last_fine_A.mm')
        b = read_mm(base_path / 'last_fine_b.mm')

    x = solve(A, b)

    with profile('solution checks'):
        check_solution(A, x, b)