"""
Implementation of the Dynamical Low-Rank Preconditioned Conjugate Gradient scheme.
"""
import numpy as np
import scipy as sp

from numpy.typing import NDArray
from typing import Optional, Union
from utils.utils import progress_bar
from algorithms.matrix_free_rsvd import MatrixFreeRSVD

from pymatting import ichol  # incomplete Cholesky
from sksparse.cholmod import cholesky  # sparse Cholesky
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import cho_factor, cho_solve, hadamard


class DynamicalLowRankPCG:
    """
    Dynamical Low-Rank Preconditioned Conjugate Gradient.
    """
    def __init__(self, mfrsvd: MatrixFreeRSVD) -> None:
        """
        Initialize the DynamicalLowRankPCG.

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

        self.residual = []  # Track residuals
        self.niter = 0  # Number of iterations to converge
        
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
            *,
            max_rank: int = 5,
            preconditioner: str = 'ic',
            truncate_tol: float = 0.01,
            X0: str = 'qr',
            max_iter: int = 250,
            rtol: float = 1e-8,
            seed: Optional[int] = None,
            verbose: bool = True
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLR-PCG scheme.
        The preconditioner must be one of: 'none', 'jacobi', 'ic', 'cholesky'

        y, NDArray          : The observed data (1D array).
        w, NDArray          : Tikhonov regularization weights (1D array).
        lambda_, float      : Tikhonov regularization parameter.
        max_iter, int       : Maximum number of iterations. 
        max_rank, int       : Max rank of the solution (dynamical step).
        preconditioner, str : Preconditioner to be used (P^{-1}).
        rtol, float         : Stopping criterion, relative residual (r0/rk).
        seed, int|None      : Seed for random number generator (for initial X).
        verbose, bool       : Print out the results and progress.
        truncate_tol, float : Truncation tolerance for the adaptive rank update.
        X0, str             : How to initialize X.

        returns: Solution vector x = vec(X) (1D array).
        """
        # Initialize X (random)
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank, X0)

        # Preconditioner (1d np.array or LinearOperator)        
        P_inv = self.get_preconditioner(w, lambda_, preconditioner)

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        Z = self.apply_P_inv(G, P_inv)
        D = -Z.copy()

        # Initial residual
        res0 = np.linalg.norm(G, 'fro')
        
        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            denom = np.sum(D * HD)
            if denom < 1e-32:
                if verbose: print(f"alpha too small ({denom}). Stopping at iter = {i}")
                break
            alpha = np.sum(G * Z) / denom

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
            denom = np.sum(G * Z)
            G = G + alpha * HD
            Z = self.apply_P_inv(G, P_inv)
            beta = np.sum(G * Z) / denom
            D = -Z + beta * D

            # Relative residual
            res = np.linalg.norm(G, 'fro')
            rel_res = res / res0
            self.residual.append(rel_res)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rel_res={rel_res:.3}]")
                break
            
            if verbose and ((i % 10 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def get_preconditioner(
            self, w: NDArray, lambda_: float, preconditioner: Optional[str]
        ) -> Union[NDArray, LinearOperator]:
        preconditioner = preconditioner.lower()
        if preconditioner == 'none':
            P_inv = np.ones(len(w))
        elif preconditioner == 'jacobi':
            P_inv = self.build_Jacobi_preconditioner(w, lambda_)
        elif preconditioner == 'cholesky':
            P_inv = self.build_cholesky_woodbury_preconditioner(w, lambda_)
        elif preconditioner == 'ic':
            P_inv = self.build_ic_woodbury_preconditioner(w, lambda_)
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner}")
        return P_inv
    
    def build_Jacobi_preconditioner(self, w: NDArray, lambda_: float) -> NDArray:
        """
        w: (N,) diagonal of W, stored as 1D array
        lambda_: float, regularization parameter
        """
        # diag(K^T M_partial K)
        A = self.U.T @ (self.M_ds @ self.U)  # (k, k)
        B = (self.S[:, None] * A) * self.S   # (k, k)
        VB = self.V @ B                      # (N, k)
        diag_KtMK = np.sum(self.V * VB, axis=1)   # (N,)

        # diag(lambda * W^T M W)
        diag_M = self.M_dx.diagonal()         # (N,)
        diag_WtMW = lambda_ * w**2 * diag_M   # (N,)

        P = diag_KtMK + diag_WtMW
        return 1.0 / P

    def build_cholesky_woodbury_preconditioner(
            self, w: NDArray, lambda_: float
        ) -> LinearOperator:
        """
        Builds a Woodbury-corrected IC preconditioner for
        H = V Sigma U^T M_partial U Sigma V^T + lambda * diag(w) M diag(w)
        """
        w_sp = sp.sparse.diags(w)
        S = lambda_ * w_sp @ self.M_dx @ w_sp
        S_csc = S.tocsc()

        A = self.U.T @ (self.M_ds @ self.U)
        L_A = np.linalg.cholesky(A)
        F = self.V * self.S[None, :] @ L_A

        # Sparse Cholesky
        factor = cholesky(S_csc)

        # Precompute these once
        Sinv_F = factor.solve_A(F)              # (N, k)
        C = np.eye(len(self.S)) + F.T @ Sinv_F  # (k, k)
        C_chol = cho_factor(C)

        def apply_P_inv(v):
            Sinv_v = factor.solve_A(v)
            correction = Sinv_F @ cho_solve(C_chol, Sinv_F.T @ v)
            return Sinv_v - correction

        N = self.V.shape[0]
        return LinearOperator((N, N), matvec=apply_P_inv)
        
    def build_ic_woodbury_preconditioner(self, w: NDArray, lambda_: float) -> LinearOperator:
        w_sp = sp.sparse.diags(w)
        S = lambda_ * w_sp @ self.M_dx @ w_sp
        S_csc = S.tocsc()

        A = self.U.T @ (self.M_ds @ self.U)
        L_A = np.linalg.cholesky(A)
        F = self.V * self.S[None, :] @ L_A

        # Incomplete Cholesky
        L_ic = ichol(S_csc)

        def apply_Sinv(v):
            return L_ic(v)    # pymatting overloads __call__ to apply L^{-T} L^{-1}

        # Precompute these once
        Sinv_F = np.column_stack([apply_Sinv(F[:, i]) for i in range(F.shape[1])])
        C = np.eye(len(self.S)) + F.T @ Sinv_F
        C_chol = cho_factor(C)

        def apply_P_inv(v):
            Sinv_v = apply_Sinv(v)
            correction = Sinv_F @ cho_solve(C_chol, Sinv_F.T @ v)
            return Sinv_v - correction

        N = self.V.shape[0]
        return LinearOperator((N, N), matvec=apply_P_inv)
    
    def initial_X(
            self, seed: Optional[int], max_rank: int, X0: str = 'svd'
        ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Generate an initial matrix X and its SVD of rank `max_rank`.
        """
        rng = np.random.default_rng(seed)

        if X0 == 'svd':
            X = rng.random((self.n, self.n)) * 1e-3
            Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)
            Sx = np.diag(sx)
            Vx = VxT.T
        
        elif X0 == 'qr':
            Ux = rng.random((self.n, self.n))
            Vx = rng.random((self.n, self.n))
            Ux, _ = np.linalg.qr(Ux)
            Vx, _ = np.linalg.qr(Vx)

            # Mimic the singular values of X ~ Uniform(0, 1):
            # sigma_1 = 0.5 * n, sigma_2, ..., sigma_n = O(sqrt(n))
            sx = np.sqrt(self.n) * rng.random(self.n)
            sx[0] = 0.5 * self.n
            sx = np.sort(sx)[::-1] * 1e-3
            Sx = np.diag(sx)

        elif X0 == 'low-rank-qr':
            Ux = rng.random((self.n, self.n))
            Vx = rng.random((self.n, self.n))
            Ux, _ = np.linalg.qr(Ux)
            Vx, _ = np.linalg.qr(Vx)

            sx = np.sqrt(self.n) * rng.random(self.n)
            sx[0] = 0.5 * self.n
            sx = np.sort(sx)[::-1] * 1e-3
            sx[max_rank:] = 0
            Sx = np.diag(sx)

        elif X0 == 'householder':
            Ux = self.fast_orthogonal(rng, max_rank)
            Vx = self.fast_orthogonal(rng, max_rank)

            sx = np.sqrt(self.n) * rng.random(self.n)
            sx[0] = 0.5 * self.n
            Sx = np.diag(np.sort(sx)[::-1] * 1e-3)

        X = Ux @ Sx @ Vx.T
        return X, Ux, Sx, Vx
    
    def fast_orthogonal(self, rng, k):
        Q = np.eye(self.n)
        for _ in range(k):
            v = rng.standard_normal(self.n)
            v /= np.linalg.norm(v)
            Q -= 2 * np.outer(v, v)
        return Q
    
    def gradient(self, X: NDArray, y: NDArray, w: NDArray, lambda_: float) -> NDArray:
            """
            Given the SVD K = U S V^T, compute the gradient of the cost
            function Phi with respect to X.
            """
            x = self.matrix_to_vec(X)
            r = self.U @ (self.S * (self.VT @ x)) - y

            grad_data = self.V @ (self.S * (self.UT @ (self.M_ds @ r)))
            grad_reg = lambda_ * (w * (self.M_dx @ (w * x)))

            return self.vec_to_matrix(grad_data + grad_reg)
    
    def apply_H(self, P: NDArray, w: NDArray, lambda_: float) -> NDArray:
        """
        Computes HP = mat[H vec(P)], where H is the Hessian of the cost function Phi.
        Here, H = K^T M_ds K + lambda * W^T M W
        """
        p = self.matrix_to_vec(P)
        Kp = self.U @ (self.S * (self.VT @ p))

        # Compute (K^T M_ds K)p and (lambda * W^T M W)p
        H_data = self.V @ (self.S * (self.UT @ (self.M_ds @ Kp)))
        H_reg = lambda_ * (w * (self.M_dx @ (w * p)))
        return self.vec_to_matrix(H_data + H_reg)
    
    def apply_P_inv(self, A: NDArray, P_inv: Union[NDArray, LinearOperator]) -> NDArray:
        """Compute mat[ P^{-1} vec(A) ]."""
        a = self.matrix_to_vec(A)
        if type(P_inv).__name__ == "_CustomLinearOperator":  # temporary hack
            P_inv_a = P_inv @ a
        else:  # assuming P_inv is a 1D array!
            P_inv_a = P_inv * a
        return self.vec_to_matrix(P_inv_a)
    
    def truncate(
            self, U1: NDArray, S: NDArray, V1: NDArray, tol: float, max_rank: int = 1
        ) -> NDArray:
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
