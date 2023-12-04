# https://docs.scipy.org/doc/scipy-1.11.2/reference/sparse.linalg.html#module-scipy.sparse.linalg
import contextlib
import logging
import time

from pathlib import Path

import numpy as np

from scipy.io import mmread
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import norm


log = logging.getLogger(__name__)

def read_mm(file_path):
    with open(file_path, 'r') as mm_file:
        data = mmread(mm_file)

    # return vectors as 1d numpy arrays
    if data.shape[1] == 1:
        return data.reshape((data.shape[0],))

    # return matrices as scipy.sparse.csr_matrix
    return sparse.csr_matrix(data)


def check(A, b, x):
    log.info(norm(b - A @ x))


@contextlib.contextmanager
def profile(label=None, log_level=logging.INFO):
    label = label if label is not None else 'context'
    tick = time.perf_counter()
    try:
        yield None
    finally:
        tock = time.perf_counter()
        log.log(level=log_level, msg=(f"{label} time: {tock-tick} seconds"))


def run():

    with profile('Loading'):
        base_path = Path('examples/bracket')
        A = read_mm(base_path / 'first_fine_A.mm')
        b = read_mm(base_path / 'first_fine_b.mm')
        inter = read_mm(base_path / 'interpolation.mm')
        restr = read_mm(base_path / 'restriction.mm')

    with profile('Coarse factorization'):
        coarse_A = sparse.csc_matrix(restr @ A @ inter)
        coarse_solve = linalg.factorized(coarse_A)

    with profile('Create preconditioner'):
        precond = linalg.LinearOperator(
            A.shape, lambda x: x - inter @ coarse_solve(restr @ x))

    diag = linalg.LinearOperator(
        A.shape, lambda x: np.divide(x, A.diagonal())
    )

    with profile('CG solve'):
        tolerance = 1e-5
        x, info = linalg.cg(A, b, tol=tolerance, atol=tolerance, M=diag)

    print(info)
    with profile('Check'):
        check(A, b, x)


if __name__ == "__main__":
    # CG First solve, diagonal preconditioner: 124 iterations, 7.769 seconds
    # CG First solve, hierarchical preconditioner: 426 iterations, 22.083 seconds
    logging.basicConfig(level=logging.DEBUG)
    run()
