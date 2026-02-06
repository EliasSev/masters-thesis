import numpy as np
import scipy as sp

from algorithms.rsvd import rsvd
from utils.utils import progress_bar

from typing import Optional
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from fenics import FunctionSpace, Function, set_log_level
set_log_level(30)


class WeightedLowRankSolver:
    def __init__(
            self, V_h: FunctionSpace, M_dx: spmatrix,
            M_ds: spmatrix, U: NDArray, S: NDArray, VT: NDArray
        ) -> None:
        self.V_h = V_h
        self.M_dx = M_dx
        self.M_ds = M_ds
        self.U = U
        self.S = S
        self.VT = VT

        # Random number generator
        self.rng = self._set_default_rng()

        # Set up vec to matrix and matrix to vec utils
        coords = V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(V_h.dim()))

    def _set_default_rng(self, seed=None):
        return np.random.default_rng(seed)

    def matrix_to_vec(self, X):
        return X.flatten()[self.dof_indices]

    def vec_to_matrix(self, x):
        return x[self.grid_indices].reshape((self.n, self.n))
    
    def solve(
            self, y: NDArray, r_s: int, w: NDArray,
            lambda_: float = 0.01, alpha: float = 1,
            max_iter: int = 100, seed: Optional[int] = None
        ) -> Function:

        # Initialize P and Q
        rng = np.random.default_rng(seed=seed)
        P, Q = self._random_P_Q(r_s=r_s, rng=rng)
        
        # Constant matrices
        U, S, VT = self.U, self.S, self.VT
        UT, V = U.T, VT.T
        M_dx, M_ds = self.M_dx, self.M_ds

        # Gradient descent on P and Q
        for i in range(max_iter):
            X = P @ Q.T
            x = self.matrix_to_vec(X)
            
            # Compute the gradient
            r = U @ (S * (VT @ x)) - y
            grad_data = V @ (S * (UT @ (M_ds @ r)))
            grad_reg = lambda_ * (w * (M_dx @ (w * x)))
            grad_Phi = grad_data + grad_reg

            # Factor gradients
            G = self.vec_to_matrix(grad_Phi)
            grad_P_Phi = G @ Q
            grad_Q_Phi = G.T @ P

            # Gradient descent
            P -= alpha * grad_P_Phi / (np.linalg.norm(grad_P_Phi) + 1e-9)
            Q -= alpha * grad_Q_Phi / (np.linalg.norm(grad_Q_Phi) + 1e-9)

            if (not i % 50) or (i + 1 == max_iter):
                progress_bar(i + 1, max_iter)
        
        # Construct final solution
        x = self.matrix_to_vec(P @ Q.T)
        f = Function(self.V_h)
        f.vector()[:] = x
        return f
    
    def _initialize_P_Q(self, initial_matrices: str, **kwargs):
        if initial_matrices == 'random':
            return self._random_P_Q(**kwargs)
        elif initial_matrices == 'tikhonov':
            return self._tikhonov_P_Q(**kwargs)
        else:
            raise ValueError(f"Unknown 'initial_matrices': '{initial_matrices}'")

    def _random_P_Q(self, r_s: int, rng: np.random.Generator, **kwargs):
        P = rng.random((self.n, r_s)) * 0.01
        Q = rng.random((self.n, r_s)) * 0.01
        return P, Q

    def _tikhonov_P_Q(self, K, y, r_s, w, lambda_,):
        # Linear operator for CG
        def matvec_A(x):
            r1 = K.T @ (self.M_ds @ (K @ x))
            Wx = w * x
            r2 = w * (self.M_dx @ Wx)
            return r1 + lambda_ * r2

        A_op = sp.sparse.linalg.LinearOperator((self.N, self.N), matvec=matvec_A)
        b = K.T @ (self.M_ds @ y)
        
        # Solve iteratively
        x_lambda, info = sp.sparse.linalg.cg(A_op, b, rtol=1e-6, maxiter=500)
        X_lambda = self.vec_to_matrix(x_lambda)

        # Randomized low-rank SVD for speed
        U, S, Vt = rsvd(X_lambda, k=r_s)
        Sigma_sqrt = np.diag(np.sqrt(S))
        P = U @ Sigma_sqrt
        Q = Vt.T @ Sigma_sqrt
        return P, Q
