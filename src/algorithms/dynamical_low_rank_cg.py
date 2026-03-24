"""
Implementation of the Dynamical Low-Rank Conjugate Gradient scheme.
"""
import numpy as np

from numpy.typing import NDArray
from typing import Optional, Union
from utils.utils import progress_bar
from algorithms.matrix_free_rsvd import MatrixFreeRSVD


class DynamicalLowRankCG:
    def __init__(self, mfrsvd: MatrixFreeRSVD) -> None:
        """
        Initialize the DynamicalLowRankCG.

        mfrsvd, MatrixFreeRSVD: A trained MatrixFreeRSVD objected.
        """
        self.V_h = mfrsvd.V_h
        self.M_dx = mfrsvd.M_dx
        self.M_ds = mfrsvd.M_ds
        self.U = mfrsvd.Uk
        self.S = mfrsvd.Sk
        self.VT = mfrsvd.VkT
        self.UT = self.U.T
        self.V = self.VT.T

        self.residual = []

        # Set up vec to matrix and matrix to vec utils
        coords = self.V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(self.V_h.dim()))

    def matrix_to_vec(self, X: NDArray) -> NDArray:
        return X.flatten()[self.dof_indices]

    def vec_to_matrix(self, x: NDArray) -> NDArray:
        return x[self.grid_indices].reshape((self.n, self.n))

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            max_iter: int = 250,
            max_rank: int = 5,
            rtol: float = 1e-8,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLR-CG scheme.

        y, NDArray              : The observed data (1D array).
        w, NDArray              : Tikhonov regularization weights (1D array).
        lambda_, float          : Tikhonov regularization parameter.
        max_iter, int           : Maximum number of iterations. 
        max_rank, int           : Max rank of the solution (dynamical step).
        rtol, float             : Stopping criterion, relative residual (r0/rk).
        seed, int|None          : Seed for random number generator (for initial X).
        verbose, bool           : Print out the results and progress.
        truncate_tol, float     : Truncation tolerance for the adaptive rank update.

        returns: Solution vector x = vec(X) (1D array).
        """
        # Initialize X (random)
        X, Ux, Sx, Vx = self.initial_X(seed)

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        # Initial residual
        res0 = np.linalg.norm(G, 'fro')
        
        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            alpha = np.sum(G * G) / np.sum(D * HD)

            # W-step
            W_star = (Ux @ Sx) + alpha * (D @ Vx)
            U_hat, _ = np.linalg.qr(np.hstack([Ux, W_star]))

            # L-ste
            L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new + alpha * (U_hat.T @ D @ V_hat)

            # Truncate back to low-rank
            Ux, Sx, Vx = self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)
            
            # Update the gradient and the search direction
            denom = np.sum(G * G)
            G = G + alpha * HD
            beta = np.sum(G * G) / denom
            D = -G + beta * D

            # Relative residual
            res = np.linalg.norm(G, 'fro')
            rel_res = res / res0
            self.residual.append(rel_res)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rel_res={rel_res:.3}]")
                break
            
            if verbose and ((i % 10 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)
        
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)
    
    def initial_X(self, seed: Union[int, None]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        rng = np.random.default_rng(seed)
        X = rng.random((self.n, self.n)) * 1e-3

        # Compute the SVD
        Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)
        Vx = VxT.T
        Sx = np.diag(sx)

        return X, Ux, Sx, Vx
    
    def apply_H(self, D: NDArray, w: NDArray, lambda_: float) -> NDArray:
        """
        Computes H * D, where H is the Hessian of the quadratic cost function.
        Here, H = K^T M_ds K + lambda * W^T M W
        """
        d = self.matrix_to_vec(D)
        Kd = self.U @ (self.S * (self.VT @ d))

        # Compute (K^T M_ds K)d and (lambda * W^T M W)d
        H_data = self.V @ (self.S * (self.UT @ (self.M_ds @ Kd)))
        H_reg = lambda_ * (w * (self.M_dx @ (w * d)))

        return self.vec_to_matrix(H_data + H_reg)
    
    def gradient(self, X: NDArray, y: NDArray, w: NDArray, lambda_: float) -> NDArray:
            """
            Given the SVD X = U S V^T, compute the gradient of the cost
            function Phi with respect to X.
            """
            x = self.matrix_to_vec(X)
            r = self.U @ (self.S * (self.VT @ x)) - y

            grad_data = self.V @ (self.S * (self.UT @ (self.M_ds @ r)))
            grad_reg = lambda_ * (w * (self.M_dx @ (w * x)))

            return self.vec_to_matrix(grad_data + grad_reg)
    
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
