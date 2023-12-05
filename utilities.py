import logging

from scipy.io import mmread
from scipy.linalg import norm
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

__all__ = ("read_mm", "check_solution")


def read_mm(file_path):
    with open(file_path, 'r') as mm_file:
        data = mmread(mm_file)

    # return vectors as 1d numpy arrays
    if data.shape[1] == 1:
        return data.reshape((data.shape[0],))

    # return matrices as scipy.sparse.csr_matrix
    return csr_matrix(data)


def check_solution(A, x, b, log_level=logging.INFO):
    resnorm = norm(b - A@x)
    log.log(level=log_level, msg=f"residual norm: {resnorm}")