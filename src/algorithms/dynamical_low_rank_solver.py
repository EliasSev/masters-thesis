"""
Implementation of the DynamicalLowRankSolver algorithm.
"""
import numpy as np

from numpy.typing import NDArray
from typing import Optional, Union
from fenics import Function
from algorithms.rsvd_solvers import MatrixFreeRSVD, MatrixFreeRSVDAdjoint
from utils.utils import progress_bar


def frobenius2(A):
    """Compute the squared Frobenius norm of A."""
    a = A.ravel()
    return np.dot(a, a)


def inner_F(A, B):
    """Compute the Frobenius inner product between A and B."""
    return np.dot(A.ravel(), B.ravel())


class DynamicalLowRankSolver:
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        """
        Initialize the DynamicalLowRankSolver.

        rsvd, MatrixFreeRSVD[Adjoint]: A trained MatrixFreeRSVD[Adjoint] instance.
        x_true, optional: True x source, to track the error while training.
        """
        self.V_h                = rsvd.V_h
        self.M_dx, self.M_ds    = rsvd.M, rsvd.M_ds
        self.U, self.S, self.VT = rsvd.U, rsvd.S, rsvd.Vt
        self.UT, self.V         = self.U.T, self.VT.T
        self.x_true             = x_true

        # Set up vec to matrix and matrix to vec utils
        coords = self.V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(self.V_h.dim()))

        self.residual = []  # Track residuals
        self.niter = 0  # Number of iterations to converge

    def matrix_to_vec(self, X: NDArray) -> NDArray:
        return X.flatten()[self.dof_indices]

    def vec_to_matrix(self, x: NDArray) -> NDArray:
        return x[self.grid_indices].reshape((self.n, self.n))

    def solve(
            self,
            y,
            w,
            lambda_ = 1e-4,
            alpha_adam = 0.1,
            rho = 0.0,
            max_iter = 5000,
            max_rank = 5,
            tol = 1e-4,
            seed = None,
            verbose = True
        ) -> Function:
        # Initialize X
        rng = np.random.default_rng(seed)
        X = rng.random((self.n, self.n)) * 1e-2

        Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)
        Vx = VxT.T
        Sx = np.diag(sx)

        # Adam Moments (same shape as X)
        m_D = np.zeros((self.n, self.n))
        v_D = np.zeros((self.n, self.n))
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        # Initial residual
        x = self.matrix_to_vec(X)
        D = self.gradient(x, y, w, lambda_, rho)[0]
        res0 = np.sqrt(frobenius2(D))

        for i in range(1, max_iter + 1):
            X = Ux @ Sx @ VxT
            x = self.matrix_to_vec(X)

            # Compute the search direction D
            D, r = self.gradient(x, y, w, lambda_, rho)

            # Adam Update for D
            m_D = beta1 * m_D + (1 - beta1) * D
            v_D = beta2 * v_D + (1 - beta2) * (D**2)
            m_hat = m_D / (1 - beta1**i)  # Bias correction
            v_hat = v_D / (1 - beta2**i)  # Adaptive gradient
            D_adam = m_hat / (np.sqrt(v_hat) + eps)

            # K-step updates K = US and leaves V fixed
            K = Ux @ Sx
            K_star = K - alpha_adam * D_adam @ Vx
            U_hat, _ = np.linalg.qr(np.hstack([Ux, K_star]))

            # L-step updates L = VS' and leaves U fixed
            L = Vx @ Sx.T
            L_star = L - alpha_adam * D_adam.T @ Ux
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step updates S at fixed (previously updated) U and V
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new - alpha_adam * (U_hat.T @ D_adam @ V_hat)

            # Truncate (adaptive)
            Ux_next, Sx_next, Vx_next = self.truncate(U_hat, S_new, V_hat, 0.01, max_rank)
            Ux, Sx, Vx = Ux_next, Sx_next, Vx_next
            VxT = Vx.T

            # Relative residual
            res = np.sqrt(frobenius2(D))
            rel_res = res / res0
            self.residual.append(rel_res)

            if rel_res < tol:
                if verbose: print(f"Converged at iter {i} [rel_res={rel_res:.3}]")
                break
            
            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        else:
            print(f"Warning: X did not converge (max_iter={max_iter} reached)")
        
        X = Ux @ Sx @ VxT
        f = Function(self.V_h)
        f.vector()[:] = self.matrix_to_vec(X)

        self.niter = i
        return f
    
    def gradient(self, x, y, w, lambda_, rho):
            """
            Given the SVD X = U S V^T, compute the gradient of the cost
            function Phi with respect to X.
            """
            # Compute the gradient D
            r = self.U @ (self.S * (self.VT @ x)) - y
            grad_data = self.V @ (self.S * (self.UT @ (self.M_ds @ r)))
            grad_reg = lambda_ * (w * (self.M_dx @ (w * x)))
            grad = grad_data + grad_reg

            # Punish negative elements
            if rho > 0:
                negative_parts = np.minimum(x, 0) # Non-zero only where X < 0
                grad_positivity = rho * negative_parts
                grad += grad_positivity

            D = self.vec_to_matrix(grad)
            return D, r
    
    def truncate(self, U1, S, V1, tol, max_rank=1):
        """
        Truncates according to tolerance
        U1: (m x k) left factor
        S:  (k x k) matrix to re-SVD (can be diagonal or full)
        V1: (n x k) right factor
        tol: scalar tolerance (same semantics as original: relative factor multiplied by norm(S))
        Returns: (U1_trunc, S_trunc, V1_trunc)
        """

        U_s, s_vals, Vh = np.linalg.svd(S, full_matrices=False)
        # convert singular values to a 1D array (s_vals already is)
        tol = tol * np.linalg.norm(S)

        # cumulative tail-sum test (Julia used sqrt(sum(...)^2) which equals abs(sum(...)))
        rmax = s_vals.size
        retained = rmax  # default keep all
        for j in range(rmax):  # 0-based index
            tail_sum = np.sum(s_vals[j:rmax])
            if abs(tail_sum) < tol:
                retained = j
                break

        if max_rank is not None:
            retained = min(retained, int(max_rank))

        # Truncation / rotate factors
        U1 = U1 @ U_s
        V1 = V1 @ Vh.T

        S_trunc = np.diag(s_vals[:retained])
        U1_trunc = U1[:, :retained]
        V1_trunc = V1[:, :retained]

        return U1_trunc, S_trunc, V1_trunc
