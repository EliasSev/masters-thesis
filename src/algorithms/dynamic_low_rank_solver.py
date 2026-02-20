"""
Implementation of the DynamicLowRankSolver algorithm.
"""
import numpy as np

from numpy.typing import NDArray
from typing import Optional
from fenics import Function
from algorithms.matrix_free_rsvd import MatrixFreeRSVD
from utils.utils import progress_bar


class DynamicLowRankSolver:
    def __init__(
            self,
            mfrsvd: MatrixFreeRSVD,
            x_true: Optional[NDArray] = None
        ) -> None:
        """
        Initialize the ProjectedLowRankSolver.

        mfrsvd, MatrixFreeRSVD: A trained MatrixFreeRSVD objected.
        x_true, optional: True x source, to track the error while training.
        """
        self.V_h = mfrsvd.V_h
        self.M_dx = mfrsvd.M_dx
        self.M_ds = mfrsvd.M_ds
        self.U = mfrsvd.Uk
        self.S = mfrsvd.Sk
        self.VT = mfrsvd.VkT
        self.UT = self.U.T
        self.V = self.VT.T
        self.x_true = x_true

        # Set up vec to matrix and matrix to vec utils
        coords = self.V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((coords[:, 0], coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(self.V_h.dim()))

        # Record history
        self._initialize_records()

    def _initialize_records(self) -> None:
        self.X_rel = []
        self.residuals = []
        if self.x_true is not None:
            self.errors = []
    
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
            seed = None
        ) -> Function:
        # Initialize X
        rng = np.random.default_rng(seed)
        X = rng.random((self.n, self.n)) * 1e-2
        X_old = X.copy()   # Track previous iteration
        X_best = X.copy()  # Track best solution in terms of residual

        Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)
        Vx = VxT.T
        Sx = np.diag(sx)

        # 2. Adam Moments (same shape as X)
        m_D = np.zeros((self.n, self.n))
        v_D = np.zeros((self.n, self.n))
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for i in range(1, max_iter + 1):
            X = Ux @ Sx @ VxT
            x = self.matrix_to_vec(X)

            # Check for convergence
            if i % 25 == 0:
                dX = np.linalg.norm(X - X_old) / (np.linalg.norm(X_old) + 1e-10)
                self.X_rel.append(dX)
                progress_bar(i, max_iter, end_text=f" (diff={dX:.1e})")
                if dX < tol:
                    print(f"\nStopping: X converged at iter {i}")
                    break
                X_old = X.copy()

            D, r = self.gradient(x, y, w, lambda_, rho)
            self.residuals.append(np.linalg.norm(r))
            self.errors.append(np.linalg.norm(self.x_true - x))

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

        else:
            print(f"Warning: X did not converge (max_iter={max_iter} reached)")
        
        X = Ux @ Sx @ VxT
        f = Function(self.V_h)
        f.vector()[:] = self.matrix_to_vec(X)
        return f
    
    def solve_cg(
            self,
            y,
            w,
            lambda_=1e-4,
            alpha_cg=0.1,
            rho=1e-4,
            max_iter=5000,
            max_rank=5,
            tol=1e-7,
            seed=None
        ) -> Function:
        # Initialize X
        rng = np.random.default_rng(seed)
        X = rng.random((self.n, self.n)) * 1e-2
        X_old = X.copy()   # Track previous iteration
        X_best = X.copy()  # Track best solution in terms of residual

        Ux, sx, VxT = np.linalg.svd(X, full_matrices=False)
        Vx = VxT.T
        Sx = np.diag(sx)

        D_prev = np.zeros((self.n, self.n))
        P_search = np.zeros((self.n, self.n))
        
        self.alphas = []
        for i in range(1, max_iter + 1):
            X = Ux @ Sx @ VxT
            x = self.matrix_to_vec(X)

            # Check for convergence
            if i % 25 == 0:
                dX = np.linalg.norm(X - X_old) / (np.linalg.norm(X_old) + 1e-10)
                self.X_rel.append(dX)
                progress_bar(i, max_iter, end_text=f" (diff={dX:.1e})")
                if dX < tol:
                    print(f"\nStopping: X converged at iter {i}")
                    break
                X_old = X.copy()
            
            # 1. Compute current raw gradient D
            D_curr, r = self.gradient(x, y, w, lambda_, rho)
            self.residuals.append(np.linalg.norm(r))
            self.errors.append(np.linalg.norm(self.x_true - x))
            
            # 2. Compute Beta (Polak-Ribière)
            if i == 1:
                P_search = D_curr.copy()
            else:
                diff_D = D_curr - D_prev
                # Dot products of flattened matrices
                numerator = np.sum(D_curr * diff_D)
                denominator = np.sum(D_prev * D_prev) + 1e-12
                beta = max(0, numerator / denominator)
                
                # Update Conjugate Search Direction
                P_search = D_curr + beta * P_search
                
                # Restart CG if the direction is no longer a descent direction
                if np.sum(P_search * D_curr) <= 0:
                    P_search = D_curr.copy()

            # Store for next iteration
            D_prev = D_curr.copy()

            # Compute "optimal" alpha
            p_vec = self.matrix_to_vec(P_search)
            Kp = self.U @ (self.S * (self.VT @ p_vec))
            #denom = np.dot(Kp, Kp) + 1e-12
            #alpha_optimal = np.dot(r, Kp) / denom
            #alpha_cg = max(0, alpha_optimal)
            Lp = w * (self.M_dx @ (w * p_vec)) # The regularizer's action on the direction

            num = np.dot(r, Kp) + lambda_ * np.dot(x, Lp)
            den = np.dot(Kp, Kp) + lambda_ * np.dot(p_vec, Lp) + 1e-12

            alpha_optimal = max(0, num / den)
            self.alphas.append(alpha_cg)
            

            # 3. Use P_search instead of D_adam in your K, L, S steps
            # K-step (Subspace Expansion)
            K_star = (Ux @ Sx) - alpha_cg * P_search @ Vx
            U_hat, _ = np.linalg.qr(np.hstack([Ux, K_star]))

            # L-step (Subspace Expansion)
            L_star = (Vx @ Sx.T) - alpha_cg * P_search.T @ Ux
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step (Projection & Update)
            P_proj = U_hat.T @ P_search @ V_hat
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new - alpha_cg * P_proj
            #S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            #S_new = S_new - alpha_cg * (U_hat.T @ P_search @ V_hat)

            # Truncate (adaptive)
            Ux_next, Sx_next, Vx_next = self.truncate(U_hat, S_new, V_hat, 0.01, max_rank)
            Ux, Sx, Vx = Ux_next, Sx_next, Vx_next
            VxT = Vx.T

        else:
            print(f"Warning: X did not converge (max_iter={max_iter} reached)")
        
        X = Ux @ Sx @ VxT
        f = Function(self.V_h)
        f.vector()[:] = self.matrix_to_vec(X)
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
