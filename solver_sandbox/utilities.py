import logging

from scipy.io import mmread
from scipy.linalg import norm
from scipy.sparse import csr_array

log = logging.getLogger(__name__)

__all__ = ("read_mm", "check_solution")


def read_mm(file_path):
    with open(file_path, 'r') as mm_file:
        data = mmread(mm_file)

    # return vectors as 1d numpy arrays
    if data.shape[1] == 1:
        return data.reshape((data.shape[0],))

    # return matrices as scipy.sparse.csr_array
    matrix = csr_array(data)
    log.debug(file_path)
    log.debug(len(matrix.nonzero()[0]))
    log.debug(matrix.shape)
    return matrix


def check_solution(A, x, b, log_level=logging.INFO):
    resnorm = norm(b - A@x)
    log.log(level=log_level, msg=f"residual norm: {resnorm}")