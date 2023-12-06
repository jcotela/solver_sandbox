# https://docs.scipy.org/doc/scipy-1.11.2/reference/sparse.linalg.html#module-scipy.sparse.linalg
import logging

from pathlib import Path

import diagonal_preconditioned_cg
import hierarchical_deflated_cg
import hierarchical_iterative_solver

from utilities import *
from performance import profile

log = logging.getLogger(__name__)


def run():

    with profile('Loading'):
        base_path = Path('examples/bracket')
        A = read_mm(base_path / 'last_fine_A.mm')
        b = read_mm(base_path / 'last_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')

    with profile('[Total diagonal precond + CG time]', logging.WARNING):
        x_cg = diagonal_preconditioned_cg.solve(A, b)

    check_solution(A, x_cg, b, log_level=logging.WARNING)

    with profile('[Total deflated CG time]', logging.WARNING):
        x_deflated = hierarchical_deflated_cg.solve(A, b, inter, restr)

    check_solution(A, x_deflated, b, log_level=logging.WARNING)

    with profile('[Total hierarchical iterative time]', logging.WARNING):
        x_hiter = hierarchical_iterative_solver.solve(A, b, inter, restr)

    check_solution(A, x_hiter, b, log_level=logging.WARNING)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run()
