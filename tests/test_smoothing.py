import unittest

import numpy as np
from scipy import sparse
from scipy.linalg import norm

from solver_sandbox.hierarchical_iterative_solver import GaussSeidel, Jacobi


def test_4x4():
    A = sparse.csr_array([
        [10.0, -1.0, 2.0, 0.0],
        [-1.0, 11.0, -1.0, 3.0],
        [2.0, -1.0, 10.0, -1.0],
        [0.0, 3.0, -1.0, 8.0],
    ])
    b = np.array([6.0, 25.0, -11.0, 15.0])
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    #solution = np.array([1.0, 2.0, -1.0, 1.0])

    return A, b, x0


def test_2x2_jacobi():
    A = sparse.csr_array(np.array([[2, 1], [5, 7]], dtype=np.float64))
    b = np.array([11, 13], dtype=np.float64)
    x0 = np.array([1.0, 1.0])
    return A, b, x0


def test_2x2_gauss_seidel():
    A = sparse.csr_array(np.array([[16, 3], [7, -11]], dtype=np.float64))
    b = np.array([11, 13], dtype=np.float64)
    x0 = np.array([1.0, 1.0])
    return A, b, x0


class TestJacobi(unittest.TestCase):

    def test_jacobi(self):
        A, b, x = test_4x4()
        jacobi = Jacobi(A, b)

        for _ in range(25):
            x = jacobi(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)

        A, b, x = test_2x2_jacobi()
        jacobi = Jacobi(A, b)

        for _ in range(35):
            x = jacobi(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)


    def test_relaxed_jacobi(self):
        A, b, x = test_4x4()
        w = 2.0/3.0
        jacobi = Jacobi(A, b, w)

        for _ in range(30):
            x = jacobi(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)

        A, b, x = test_2x2_jacobi()
        jacobi = Jacobi(A, b, w)

        for _ in range(60):
            x = jacobi(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)


class TestGaussSeidel(unittest.TestCase):

    def test_gauss_seidel(self):
        A, b, x = test_4x4()
        gs = GaussSeidel(A, b)

        for _ in range(25):
            x = gs(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)

        A, b, x = test_2x2_gauss_seidel()
        gs = GaussSeidel(A, b)

        for _ in range(25):
            x = gs(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)


    def test_sor(self):
        A, b, x = test_4x4()
        w = 1.2
        gs = GaussSeidel(A, b, w)

        for _ in range(25):
            x = gs(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)

        A, b, x = test_2x2_gauss_seidel()
        gs = GaussSeidel(A, b, w)

        for _ in range(25):
            x = gs(x)

        self.assertLessEqual(norm(b - A@x), 1e-6)