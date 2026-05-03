"""
Implementation of CG / low-rank solvers: ConjugateGradient, DynamicalLowRankApproximation,
DynamicalLowRankCG, DynamicalLowRankPCG, RiemannianGradientDescent, and
RiemannianConjugateGradient.
"""

import numpy as np
import scipy as sp

from time import time
from numpy.typing import NDArray
from typing import Optional, Union
from abc import ABC, abstractmethod
from utils.utils import progress_bar
from algorithms.rsvd_solvers import MatrixFreeRSVD, MatrixFreeRSVDAdjoint

from pymatting import ichol  # incomplete Cholesky
from sksparse.cholmod import cholesky  # sparse Cholesky
from scipy.sparse.linalg import LinearOperator, spsolve_triangular, factorized
from scipy.linalg import cho_factor, cho_solve


def frobenius2(A):
    """Compute the squared Frobenius norm of A."""
    a = A.ravel()
    return np.dot(a, a)


def inner_F(A, B):
    """Compute the Frobenius inner product between A and B."""
    return np.dot(A.ravel(), B.ravel())


class CGSolver(ABC):
    """
    Base class for the class of CG solver:
        - Conjugate Gradient (matrix implementation)
        - DynamicalLowRankApproximation
        - DynamicalLowRankCG
        - DynamicalLowRankPCG
    """
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        """
        Initialize the DynamicalLowRankPCG.

        rsvd, MatrixFreeRSVD[Adjoint]: A trained MatrixFreeRSVD[Adjoint] objected.
        """
        self.V_h                = rsvd.V_h
        self.M_dx, self.M_ds    = rsvd.M, rsvd.M_ds
        self.U, self.S, self.VT = rsvd.U, rsvd.S, rsvd.Vt
        self.UT, self.V         = self.U.T, self.VT.T

        self._x_true = None
        self._X_true = None

        self.error = []     # Track the error
        self.residual = []  # Track residuals
        self.niter = 0      # Number of iterations to converge
        
        # Set up vec to matrix and matrix to vec utils
        coords = self.V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(self.V_h.dim()))
        
        # Must be done after grid_indices is set up
        if x_true is not None:
            self.x_true = x_true
    
    @property
    def x_true(self) -> NDArray:
        if self._x_true is None:
            raise ValueError("'x_true' is not set!")
        return self._x_true
    
    @property
    def X_true(self) -> NDArray:
        if self._X_true is None:
            raise ValueError("'X_true' is not set!")
        return self._X_true
    
    @x_true.setter
    def x_true(self, value: NDArray) -> None:
        self._x_true = value
        self._X_true = self.vec_to_matrix(value)

    @X_true.setter
    def X_true(self, value: NDArray) -> None:
        self._x_true = self.matrix_to_vec(value)
        self._X_true = value

    def matrix_to_vec(self, X: NDArray) -> NDArray:
        return X.flatten()[self.dof_indices]

    def vec_to_matrix(self, x: NDArray) -> NDArray:
        return x[self.grid_indices].reshape((self.n, self.n))
    
    @abstractmethod
    def solve(self):
        """Solver method to be implemented by subclasses."""
        pass
    
    def initial_X(
            self, seed: Optional[int], max_rank: int, X0: Union[str, NDArray]
        ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Generate an initial matrix X and its SVD of rank `max_rank`.
        """
        rng = np.random.default_rng(seed)

        # Custom 'X0' passed in by user
        if isinstance(X0, np.ndarray):
            Ux, sx, VxT = np.linalg.svd(X0, full_matrices=False)
            Sx = np.diag(sx)
            Vx = VxT.T

        elif isinstance(X0, str):
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
            
            else:
                raise ValueError(f"Invalid 'X0': '{X0}'")

        else:
            raise ValueError(f"Invalid 'X0' type {type(X0)}")

        X = Ux @ Sx @ Vx.T
        return X, Ux, Sx, Vx
    
    def fast_orthogonal(self, rng, k):
        """
        Apply Householder transformations form an orthogonal Q.
        """
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
    
    def project_tangent(
            self, A: NDArray, Ux: NDArray, Vx: NDArray
        ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Project A onto the tangent space T_X M_r at X = Ux Sx Vx^T.

        The tangent vector decomposes as
            xi = Ux M Vx^T + Q_U R_U Vx^T + Ux R_V^T Q_V^T,
        with M = Ux^T A Vx, Q_U R_U = (I - Ux Ux^T) A Vx, and
        Q_V R_V = (I - Vx Vx^T) A^T Ux. Q_U has columns orthogonal to Ux,
        Q_V has columns orthogonal to Vx.

        Returns xi as an explicit (n, n) matrix together with the structured
        factors so the retraction can be done via a small (2r x 2r) SVD.
        """
        M_mat = Ux.T @ A @ Vx                    # (r, r)
        U_p = A @ Vx - Ux @ M_mat                # (n, r), columns orthogonal to Ux
        V_p = A.T @ Ux - Vx @ M_mat.T            # (n, r), columns orthogonal to Vx
        Q_U, R_U = np.linalg.qr(U_p)
        Q_V, R_V = np.linalg.qr(V_p)
        xi = Ux @ M_mat @ Vx.T + Q_U @ R_U @ Vx.T + Ux @ R_V.T @ Q_V.T
        return xi, M_mat, Q_U, R_U, Q_V, R_V

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

    # -------- Preconditioners --------
    # Operate on the Euclidean Hessian H = K^T M_partial K + lambda * W^T M W,
    # exposed as a 1D diagonal (Jacobi) or LinearOperator. Used by
    # DynamicalLowRankPCG and RiemannianConjugateGradient.

    def get_preconditioner(
            self, w: NDArray, lambda_: float, preconditioner: str
        ) -> Union[NDArray, LinearOperator]:
        preconditioner = preconditioner.lower()
        valid_preconditioners = ('none', 'jacobi', 'ssor', 'ic', 'ic-woodbury', 'perfect')

        if preconditioner == 'none':
            P_inv = np.ones(len(w))
        elif preconditioner == 'jacobi':
            P_inv = self._jacobi_preconditioner(w, lambda_)
        elif preconditioner == 'ssor':
            P_inv = self._ssor_preconditioner(w, lambda_)
        elif preconditioner == 'ic':
            P_inv = self._ic_preconditioner(w, lambda_)
        elif preconditioner == 'ic-woodbury':
            P_inv = self._ic_woodbury_preconditioner(w, lambda_)
        elif preconditioner == 'perfect':
            P_inv = self._perfect_preconditioner(w, lambda_)
        else:
            raise ValueError(f"Unknown preconditioner '{preconditioner}'. Use one of: {valid_preconditioners}")
        return P_inv

    def _jacobi_preconditioner(self, w: NDArray, lambda_: float) -> NDArray:
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

    def _ssor_preconditioner(
            self, w: NDArray, lambda_: float, omega: float = 1.0
        ) -> LinearOperator:
        """
        SSOR preconditioner applied to the sparse part S = lambda * W^T M W.
        For omega=1 this reduces to Symmetric Gauss-Seidel.
        P = (D/omega + L) (D/omega)^{-1} (D/omega + L^T),
        where D = diag(S) and L is the strictly lower triangular part of S.
        Applied as P^{-1} v via one forward and one backward triangular solve.
        """
        w_sp = sp.sparse.diags(w)
        S = lambda_ * w_sp @ self.M_dx @ w_sp

        D_over_omega = S.diagonal() / omega
        D_diag = sp.sparse.diags(D_over_omega)
        L = sp.sparse.tril(S, k=-1)   # strictly lower triangular

        lower = (D_diag + L).tocsc()
        upper = lower.T.tocsc()

        solve_lower = factorized(lower)
        solve_upper = factorized(upper)

        def apply_P_inv(v):
            y = solve_lower(v)
            z = D_over_omega * y
            return solve_upper(z)

        N = self.V.shape[0]
        return LinearOperator((N, N), matvec=apply_P_inv)

    def _perfect_preconditioner(
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

    def _ic_preconditioner(self, w: NDArray, lambda_: float) -> LinearOperator:
        """
        Builds a simple IC preconditioner using only the sparse part of H:
        P approx S = lambda * W^T M W approx L L^T, applied as P^{-1} = L^{-T} L^{-1}.
        The low-rank part K^T M_partial K is ignored — no Woodbury correction.
        """
        w_sp = sp.sparse.diags(w)
        S = lambda_ * w_sp @ self.M_dx @ w_sp
        S_csc = S.tocsc()

        L_ic = ichol(S_csc)

        def apply_P_inv(v):
            return L_ic(v)  # pymatting overloads __call__ to apply L^{-T} L^{-1}

        N = self.V.shape[0]
        return LinearOperator((N, N), matvec=apply_P_inv)

    def _ic_woodbury_preconditioner(
            self, w: NDArray, lambda_: float
        ) -> LinearOperator:
        """
        Woodbury-corrected IC preconditioner using the direct factored form:
          H = S + U_tilde C U_tilde^T,
        where U_tilde = V_k Sigma_k (N x k) and C = U_k^T M_partial U_k (k x k).
        Applies the Woodbury identity directly without factoring C, giving:
          P^{-1} v = S^{-1}v - (S^{-1} U_tilde)(C^{-1} + U_tilde^T S^{-1} U_tilde)^{-1}(U_tilde^T S^{-1} v)
        where S^{-1} is approximated via incomplete Cholesky.
        """
        w_sp = sp.sparse.diags(w)
        S = lambda_ * w_sp @ self.M_dx @ w_sp
        S_csc = S.tocsc()

        U_tilde = self.V * self.S[None, :]              # (N, k): V_k Sigma_k
        C = self.U.T @ (self.M_ds @ self.U)             # (k, k): U_k^T M_partial U_k
        C_inv = np.linalg.inv(C)

        L_ic = ichol(S_csc)

        def apply_Sinv(v):
            return L_ic(v)

        Sinv_Ut = np.column_stack(
            [apply_Sinv(U_tilde[:, i]) for i in range(U_tilde.shape[1])]
        )                                                # (N, k): S^{-1} U_tilde
        W_mat = C_inv + U_tilde.T @ Sinv_Ut             # (k, k): Woodbury correction matrix
        W_chol = cho_factor(W_mat)

        def apply_P_inv(v):
            Sinv_v = apply_Sinv(v)
            correction = Sinv_Ut @ cho_solve(W_chol, U_tilde.T @ Sinv_v)
            return Sinv_v - correction

        N = self.V.shape[0]
        return LinearOperator((N, N), matvec=apply_P_inv)

    def apply_P_inv(self, A: NDArray, P_inv: Union[NDArray, LinearOperator]) -> NDArray:
        """Compute mat[ P^{-1} vec(A) ]."""
        a = self.matrix_to_vec(A)
        if isinstance(P_inv, LinearOperator):
            return self.vec_to_matrix(P_inv @ a)
        else:
            return self.vec_to_matrix(P_inv * a)


class ConjugateGradient(CGSolver):
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        super().__init__(rsvd, x_true)

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            X0: str = 'qr',
            X0_rank: int = 3,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using a standard CG scheme.

        y, NDArray     : The observed data (1D array).
        w, NDArray     : Tikhonov regularization weights (1D array).
        lambda_, float : Tikhonov regularization parameter.
        max_iter, int  : Maximum number of iterations. 
        X0, str        : How to initialize X.
        X0_rank, int   : The rank of X0 (only used if X0 = 'qr-low-rank' or 'householder')
        rtol, float    : Stopping criterion, relative residual (r0/rk).
        etol, float    : Alternative stopping criterion, relative error (e0/ek).
        seed, int|None : Seed for random number generator (for initial X).
        verbose, bool  : Print out the results and progress.

        returns: Solution vector x = vec(X) (1D array).
        """
        # Temp fix (lambda should be squared directly in the implementation!)
        lambda_ = lambda_**2

        # Initialize X (random)
        X = self.initial_X(seed, max_rank=X0_rank, X0=X0)[0]

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()
        
        # Initial residual
        res0 = np.sqrt(frobenius2(G))
        
        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            alpha = frobenius2(G) / inner_F(D, HD)

            # Update X
            X = X + alpha * D
            
            # Update the gradient and the search direction
            denom = frobenius2(G)
            G = G + alpha * HD
            beta = frobenius2(G) / denom
            D = -G + beta * D

            # Relative residual
            res = np.sqrt(frobenius2(G))
            rel_res = res / res0
            self.residual.append(rel_res)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rel_res={rel_res:.3}]")
                break
            
            if verbose and ((i % 10 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)
        
        self.niter = i
        return self.matrix_to_vec(X)
    

class DynamicalLowRankApproximation(CGSolver):
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        super().__init__(rsvd, x_true)

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            method: str = 'adam',
            alpha: float = 0.1,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLRA scheme
        with a selectable step rule.

        y, NDArray          : The observed data (1D array).
        w, NDArray          : Tikhonov regularization weights (1D array).
        lambda_, float      : Tikhonov regularization parameter.
        method, str         : Step rule, one of:
                             'adam'  - adaptive step via Adam moments,
                             'fixed' - constant step size along -G,
                             'sd'    - steepest descent with exact line search.
        alpha, float        : Step size for 'fixed'; base learning rate for 'adam'.
                             Ignored when method='sd'.
        beta1, beta2, float : Adam moment decay rates. Used only when method='adam'.
        eps, float          : Adam denominator stabilizer. Used only when method='adam'.
        X0, str             : How to initialize X.
        max_rank, int       : Max rank of the solution (dynamical step).
        max_iter, int       : Maximum number of iterations.
        rtol, float         : Stopping criterion, relative residual (r0/rk).
        etol, float         : Alternative stopping criterion, relative error (e0/ek).
        seed, int|None      : Seed for random number generator (for initial X).
        verbose, bool       : Print out the results and progress.
        truncate_tol, float : Truncation tolerance for the adaptive rank update.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        common = dict(
            X0=X0, max_rank=max_rank, max_iter=max_iter, rtol=rtol, etol=etol,
            seed=seed, verbose=verbose, truncate_tol=truncate_tol,
        )
        method = method.lower()
        if method == 'adam':
            return self._solve_adam(
                y, w, lambda_, alpha=alpha,
                beta1=beta1, beta2=beta2, eps=eps, **common,
            )
        elif method == 'fixed':
            return self._solve_fixed(y, w, lambda_, alpha=alpha, **common)
        elif method == 'sd':
            return self._solve_sd(y, w, lambda_, **common)
        else:
            raise ValueError(
                f"Unknown method: '{method}'. Use 'adam', 'fixed', or 'sd'."
            )

    def _solve_adam(
            self, y, w, lambda_, *,
            alpha, beta1, beta2, eps,
            X0, max_rank, max_iter, rtol, etol, seed, verbose, truncate_tol,
        ):
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        m_D = np.zeros((self.n, self.n))
        v_D = np.zeros((self.n, self.n))

        for i in range(1, max_iter + 1):
            # Adam update for D
            m_D = beta1 * m_D + (1 - beta1) * D
            v_D = beta2 * v_D + (1 - beta2) * (D**2)
            m_hat = m_D / (1 - beta1**i)
            v_hat = v_D / (1 - beta2**i)
            D_adam = m_hat / (np.sqrt(v_hat) + eps)

            Ux, Sx, Vx = self._dlra_step(Ux, Sx, Vx, D_adam, alpha, truncate_tol, max_rank)

            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            D = -G.copy()

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def _solve_fixed(
            self, y, w, lambda_, *,
            alpha,
            X0, max_rank, max_iter, rtol, etol, seed, verbose, truncate_tol,
        ):
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        for i in range(1, max_iter + 1):
            Ux, Sx, Vx = self._dlra_step(Ux, Sx, Vx, D, alpha, truncate_tol, max_rank)

            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            D = -G.copy()

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def _solve_sd(
            self, y, w, lambda_, *,
            X0, max_rank, max_iter, rtol, etol, seed, verbose, truncate_tol,
        ):
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        for i in range(1, max_iter + 1):
            # Exact line search along D = -G:  alpha = ||G||^2 / <D, HD>
            HD = self.apply_H(D, w, lambda_)
            alpha = frobenius2(G) / inner_F(D, HD)

            Ux, Sx, Vx = self._dlra_step(Ux, Sx, Vx, D, alpha, truncate_tol, max_rank)

            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            D = -G.copy()

            if self._track_and_check(G, X, res0, err0, rtol, etol, i, max_iter, verbose):
                break

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def _dlra_step(self, Ux, Sx, Vx, D, alpha, truncate_tol, max_rank):
        """One W/L/S DLRA step along direction D with step size alpha, then truncate."""
        # W-step
        W_star = (Ux @ Sx) + alpha * (D @ Vx)
        U_hat, _ = np.linalg.qr(np.hstack([Ux, W_star]))

        # L-step
        L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
        V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

        # S-step
        S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
        S_new = S_new + alpha * (U_hat.T @ D @ V_hat)

        return self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)

    def _track_and_check(self, G, X, res0, err0, rtol, etol, i, max_iter, verbose):
        """Record residual/error, check convergence, print progress. Returns True if done."""
        rel_res = np.sqrt(frobenius2(G)) / res0
        self.residual.append(rel_res)

        rel_err = np.sqrt(frobenius2(X - self.X_true)) / err0
        self.error.append(rel_err)

        if rel_res < rtol:
            if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
            return True
        
        if etol is not None:
            if rel_err < etol:
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_res={rel_err:.3}]")
                return True
        
        if verbose and ((i % 100 == 0) or (i == max_iter)):
            progress_bar(i, max_iter)
        return False


class DynamicalLowRankCG(CGSolver):
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        super().__init__(rsvd, x_true)
        self.setup_time = 0
        self.CG_time = 0
        self.WLS_time = 0
        self.truncate_time = 0

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
            restart_every: Optional[int] = None,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLR-CG scheme.

        y, NDArray              : The observed data (1D array).
        w, NDArray              : Tikhonov regularization weights (1D array).
        lambda_, float          : Tikhonov regularization parameter.
        max_iter, int           : Maximum number of iterations.
        max_rank, int           : Max rank of the solution (dynamical step).
        rtol, float             : Stopping criterion, relative residual (r0/rk).
        etol, float             : Alternative stopping criterion, relative error (e0/ek).
        seed, int|None          : Seed for random number generator (for initial X).
        verbose, bool           : Print out the results and progress.
        truncate_tol, float     : Truncation tolerance for the adaptive rank update.
        restart_every, int|None : If set, recompute the true gradient at the truncated X
                                  and reset D = -G every this many iterations, correcting
                                  drift between the CG recurrence and the truncated iterate.
                                  If None, no restarting is performed.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]
        
        # Initialize X (random)
        self.setup_time = self.CG_time = self.WLS_time = 0
        t0 = time()
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()
        self.setup_time = time() - t0

        # Initial residual
        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        for i in range(1, max_iter + 1):
            # Step size
            t0 = time()
            HD = self.apply_H(D, w, lambda_)
            alpha = frobenius2(G) / inner_F(D, HD)
            self.CG_time += 0

            # W-step
            t0 = time()
            W_star = (Ux @ Sx) + alpha * (D @ Vx)
            U_hat, _ = np.linalg.qr(np.hstack([Ux, W_star]))

            # L-step
            L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new + alpha * (U_hat.T @ D @ V_hat)
            self.WLS_time += time() - t0

            # Truncate back to low-rank
            t0 = time()
            Ux, Sx, Vx = self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)
            self.truncate_time += time() - t0

            # Update the gradient and the search direction
            t0 = time()
            if restart_every is not None and i % restart_every == 0:
                X = Ux @ Sx @ Vx.T
                G = self.gradient(X, y, w, lambda_)
                D = -G.copy()
            else:
                denom = frobenius2(G)
                G = G + alpha * HD
                beta = frobenius2(G) / denom
                D = -G + beta * D
            self.CG_time += time() - t0

            # Relative residual
            res = np.sqrt(frobenius2(G))
            rel_res = res / res0
            self.residual.append(rel_res)

            # Relative error
            err = np.sqrt(frobenius2(Ux @ Sx @ Vx.T - self.X_true))
            rel_err = err / err0
            self.error.append(rel_err)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
                break
            
            if etol is not None:
                if rel_err < etol:
                    if verbose: print(f"Converged at iter {i} [etol criteria: rel_res={rel_err:.3}]")
                    break

            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)


class DynamicalLowRankPCG(CGSolver):
    """
    Dynamical Low-Rank Preconditioned Conjugate Gradient.
    """
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        super().__init__(rsvd, x_true)

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            max_rank: int = 5,
            *,
            preconditioner: str = 'ic',
            truncate_tol: float = 0.01,
            X0: str = 'qr',
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: float = 0.0,
            seed: Optional[int] = None,
            verbose: bool = True,
            restart_every: Optional[int] = None,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using the DLR-PCG scheme.
        The preconditioner must be one of: 'none', 'jacobi', 'ssor', 'ic', 'ic-woodbury', 'perfect'

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
        restart_every, int|None : If set, recompute the true gradient at the truncated X
                                  and reset D = -G every this many iterations.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2  # Temporary correction to match thesis
        self.error, self.residual = [1.0], [1.0]  # Initial relative err/res

        # Initialize X (random)
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank, X0)

        # Preconditioner (1d np.array or LinearOperator)        
        P_inv = self.get_preconditioner(w, lambda_, preconditioner)

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        Z = self.apply_P_inv(G, P_inv)
        D = -Z.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))
        
        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            denom = inner_F(D, HD)
            alpha = inner_F(G, Z) / denom

            # W-step
            W_star = (Ux @ Sx) + alpha * (D @ Vx)
            U_hat, _ = np.linalg.qr(np.hstack([Ux, W_star]))

            # L-step
            L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new + alpha * (U_hat.T @ D @ V_hat)

            # Truncate back to low-rank
            Ux, Sx, Vx = self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)
            
            # Update the gradient and the search direction
            if restart_every is not None and i % restart_every == 0:
                X = Ux @ Sx @ Vx.T
                G = self.gradient(X, y, w, lambda_)
                Z = self.apply_P_inv(G, P_inv)
                D = -Z.copy()
            else:
                denom = inner_F(G, Z)
                G = G + alpha * HD
                Z = self.apply_P_inv(G, P_inv)
                beta = inner_F(G, Z) / denom
                D = -Z + beta * D

            # Relative residual
            res = np.sqrt(frobenius2(G))
            rel_res = res / res0
            self.residual.append(rel_res)

            # Relative error
            err = np.sqrt(frobenius2(Ux @ Sx @ Vx.T - self.X_true))
            rel_err = err / err0
            self.error.append(rel_err)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rel_res={rel_res:.3}]")
                break

            if rel_err < etol:
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_res={rel_err:.3}]")
                break
            
            if verbose and ((i % 10 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)


class RiemannianGradientDescent(CGSolver):
    """
    Riemannian Gradient Descent on the fixed-rank manifold M_r.

    Each step projects the Euclidean gradient G = grad Phi(X) onto T_X M_r,
    takes a step in the tangent space, and retracts back to M_r via a small
    (2r x 2r) SVD on the structured representation of X - alpha * xi.
    """
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        super().__init__(rsvd, x_true)

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            method: str = 'sd',
            alpha: float = 1.0,
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} via Riemannian Gradient Descent on the
        manifold of rank-`max_rank` matrices.

        y, NDArray          : The observed data (1D array).
        w, NDArray          : Tikhonov regularization weights (1D array).
        lambda_, float      : Tikhonov regularization parameter.
        method, str         : Step rule, one of:
                              'sd'    - exact line search (alpha = |xi|^2/<xi,H xi>),
                              'fixed' - constant step size `alpha`.
        alpha, float        : Step size for 'fixed'. Ignored when method='sd'.
        X0, str             : How to initialize X.
        max_rank, int       : Manifold rank r.
        max_iter, int       : Maximum number of iterations.
        rtol, float         : Stopping criterion on the relative Riemannian
                              gradient norm |xi_k| / |xi_0|.
        etol, float         : Alternative stopping criterion, relative error.
        seed, int|None      : Seed for random number generator (for initial X).
        verbose, bool       : Print out the results and progress.
        truncate_tol, float : Truncation tolerance used by the retraction.

        returns: Solution vector x = vec(X) (1D array).

        Note: residual list tracks the *Riemannian* gradient norm — this is
        the natural stationarity measure on the manifold, and unlike the
        Euclidean gradient norm it actually goes to zero at a critical point
        of Phi restricted to M_r.
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Initialize on the manifold: start at the rank-`max_rank` truncation of X0
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)
        Ux, Sx, Vx = self.truncate(Ux, Sx, Vx, truncate_tol, max_rank)
        X = Ux @ Sx @ Vx.T

        # Initial Riemannian gradient
        G = self.gradient(X, y, w, lambda_)
        xi, M_mat, Q_U, R_U, Q_V, R_V = self.project_tangent(G, Ux, Vx)

        res0 = np.sqrt(frobenius2(xi))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        method = method.lower()
        if method not in ('sd', 'fixed'):
            raise ValueError(f"Unknown method: '{method}'. Use 'sd' or 'fixed'.")

        for i in range(1, max_iter + 1):
            # Step size
            if method == 'sd':
                Hxi = self.apply_H(xi, w, lambda_)
                alpha_i = frobenius2(xi) / inner_F(xi, Hxi)
            else:
                alpha_i = alpha

            # Retraction: X - alpha*xi = [Ux, Q_U] B [Vx, Q_V]^T, then truncate
            r = Sx.shape[0]
            B = np.block([
                [Sx - alpha_i * M_mat,  -alpha_i * R_V.T   ],
                [-alpha_i * R_U,         np.zeros((r, r))  ],
            ])
            U_aug = np.hstack([Ux, Q_U])
            V_aug = np.hstack([Vx, Q_V])
            Ux, Sx, Vx = self.truncate(U_aug, B, V_aug, truncate_tol, max_rank)

            # Recompute Euclidean and Riemannian gradient at the retracted point
            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            xi, M_mat, Q_U, R_U, Q_V, R_V = self.project_tangent(G, Ux, Vx)

            # Track convergence
            rel_res = np.sqrt(frobenius2(xi)) / res0
            self.residual.append(rel_res)

            rel_err = np.sqrt(frobenius2(X - self.X_true)) / err0
            self.error.append(rel_err)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
                break

            if etol is not None and rel_err < etol:
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_err={rel_err:.3}]")
                break

            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)


class RiemannianConjugateGradient(CGSolver):
    """
    Riemannian (Preconditioned) Conjugate Gradient on the fixed-rank manifold M_r.

    Search direction
        D_{k+1} = -xi_tilde_{k+1} + beta_{k+1} * T_k(D_k),
    where xi_tilde = P_T(P^{-1} G) is the (preconditioned) Riemannian gradient
    and T_k(D) = P_{T_{X_{k+1}}}(D) is the projection-based vector transport.
    Step size is exact line search alpha = -<G, D>/<D, H D> (well-defined since
    Phi is quadratic).

    With `preconditioner='none'` the preconditioned gradient xi_tilde reduces to
    the plain Riemannian gradient xi, recovering standard RCG. With any other
    preconditioner P^{-1} ~ H^{-1}, we get RPCG. The available preconditioners
    are the same as for DynamicalLowRankPCG: 'none', 'jacobi', 'ssor', 'ic',
    'ic-woodbury', 'perfect'.

    The beta formulas use the Riemannian-PCG analogues:
        FR : beta = <xi_new,    xi_tilde_new> / <xi_old, xi_tilde_old>
        PR : beta = <xi_tilde_new, xi_new - T(xi_old)> / <xi_old, xi_tilde_old>
        PR+: beta = max(0, beta_PR)
    These reduce to the un-preconditioned formulas when xi_tilde = xi.
    """
    def __init__(
            self,
            rsvd: Union[MatrixFreeRSVD, MatrixFreeRSVDAdjoint],
            x_true: Optional[NDArray] = None
        ) -> None:
        super().__init__(rsvd, x_true)

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            beta_rule: str = 'fr',
            preconditioner: str = 'none',
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
            restart_every: Optional[int] = None,
        ) -> NDArray:
        """
        Solve min{Phi(X; y, w)} via Riemannian (Preconditioned) Conjugate
        Gradient on the manifold of rank-`max_rank` matrices.

        y, NDArray              : The observed data (1D array).
        w, NDArray              : Tikhonov regularization weights (1D array).
        lambda_, float          : Tikhonov regularization parameter.
        beta_rule, str          : Conjugacy rule, one of 'fr', 'pr', 'pr+'.
        preconditioner, str     : Preconditioner P^{-1}, one of 'none', 'jacobi',
                                  'ssor', 'ic', 'ic-woodbury', 'perfect'.
        X0, str                 : How to initialize X.
        max_rank, int           : Manifold rank r.
        max_iter, int           : Maximum number of iterations.
        rtol, float             : Stopping criterion on the relative Riemannian
                                  gradient norm |xi_k| / |xi_0|.
        etol, float             : Alternative stopping criterion, relative error.
        seed, int|None          : Seed for random number generator (for initial X).
        verbose, bool           : Print out the results and progress.
        truncate_tol, float     : Truncation tolerance used by the retraction.
        restart_every, int|None : If set, reset D = -xi_tilde every this many
                                  iterations to recover from drift.

        returns: Solution vector x = vec(X) (1D array).
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        beta_rule = beta_rule.lower()
        if beta_rule not in ('fr', 'pr', 'pr+'):
            raise ValueError(f"Unknown beta_rule: '{beta_rule}'. Use 'fr', 'pr', or 'pr+'.")

        # Build preconditioner once (P^{-1} is constant across iterations)
        P_inv = self.get_preconditioner(w, lambda_, preconditioner)

        # Initialize on the manifold
        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)
        Ux, Sx, Vx = self.truncate(Ux, Sx, Vx, truncate_tol, max_rank)
        X = Ux @ Sx @ Vx.T

        # Initial Riemannian gradient (xi) and preconditioned Riemannian
        # gradient (xi_p). With preconditioner='none', xi_p == xi.
        G = self.gradient(X, y, w, lambda_)
        xi,   *_ = self.project_tangent(G, Ux, Vx)
        xi_p, *_ = self.project_tangent(self.apply_P_inv(G, P_inv), Ux, Vx)
        D = -xi_p.copy()

        res0 = np.sqrt(frobenius2(xi))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        for i in range(1, max_iter + 1):
            # Decompose D into structured form at the *current* iterate (D is
            # already tangent at X_k, so this projection is the identity but
            # gives us the factored representation needed for cheap retraction).
            _, M_D, Q_U_D, R_U_D, Q_V_D, R_V_D = self.project_tangent(D, Ux, Vx)

            # Exact line search on the quadratic along D
            HD = self.apply_H(D, w, lambda_)
            alpha_i = -inner_F(G, D) / inner_F(D, HD)

            # PCG denominator <xi_old, xi_tilde_old>; cache before xi/xi_p
            # are overwritten at the new iterate.
            denom_old = inner_F(xi, xi_p)

            # Retraction: X + alpha*D = [Ux, Q_U_D] B [Vx, Q_V_D]^T, then truncate
            r = Sx.shape[0]
            B = np.block([
                [Sx + alpha_i * M_D,    alpha_i * R_V_D.T ],
                [alpha_i * R_U_D,       np.zeros((r, r))  ],
            ])
            U_aug = np.hstack([Ux, Q_U_D])
            V_aug = np.hstack([Vx, Q_V_D])
            Ux, Sx, Vx = self.truncate(U_aug, B, V_aug, truncate_tol, max_rank)

            # Gradient and preconditioned Riemannian gradient at the new iterate
            X = Ux @ Sx @ Vx.T
            G = self.gradient(X, y, w, lambda_)
            xi_new,   *_ = self.project_tangent(G, Ux, Vx)
            xi_p_new, *_ = self.project_tangent(self.apply_P_inv(G, P_inv), Ux, Vx)

            # Update direction (with optional periodic restart)
            if restart_every is not None and i % restart_every == 0:
                D = -xi_p_new.copy()
            else:
                # Vector transport: project old D (and old xi if PR/PR+ needs it)
                D_transp, *_ = self.project_tangent(D, Ux, Vx)

                if beta_rule == 'fr':
                    beta = inner_F(xi_new, xi_p_new) / denom_old
                else:  # 'pr' or 'pr+'
                    xi_transp, *_ = self.project_tangent(xi, Ux, Vx)
                    beta = inner_F(xi_p_new, xi_new - xi_transp) / denom_old
                    if beta_rule == 'pr+':
                        beta = max(0.0, beta)

                D = -xi_p_new + beta * D_transp

            xi   = xi_new
            xi_p = xi_p_new

            # Track convergence (Riemannian gradient norm and relative error)
            rel_res = np.sqrt(frobenius2(xi)) / res0
            self.residual.append(rel_res)

            rel_err = np.sqrt(frobenius2(X - self.X_true)) / err0
            self.error.append(rel_err)

            if rel_res < rtol:
                if verbose: print(f"Converged at iter {i} [rtol criteria: rel_res={rel_res:.3}]")
                break

            if etol is not None and rel_err < etol:
                if verbose: print(f"Converged at iter {i} [etol criteria: rel_err={rel_err:.3}]")
                break

            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)
