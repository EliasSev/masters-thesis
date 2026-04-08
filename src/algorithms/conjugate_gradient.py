"""
Implementation of a matrix version of the Conjugate Gradient scheme.
"""
import numpy as np

from typing import Optional
from numpy.typing import NDArray
from utils.utils import progress_bar
from algorithms.matrix_free_rsvd import MatrixFreeRSVD


class ConjugateGradient:
    def __init__(self, mfrsvd: MatrixFreeRSVD) -> None:
        """
        Initialize the ConjugateGradient.

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
        self.niter = 0 

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
            X0: str = 'qr',
            X0_rank: int = 3,
            max_iter: int = 250,
            rtol: float = 1e-8,
            seed: Optional[int] = None,
            verbose: bool = True,
        ):
        """
        Solve min{Phi(X; y, w)} with given lambda_ and max_rank using a standard CG scheme.

        y, NDArray              : The observed data (1D array).
        w, NDArray              : Tikhonov regularization weights (1D array).
        lambda_, float          : Tikhonov regularization parameter.
        max_iter, int           : Maximum number of iterations. 
        X0, str                 : How to initialize X.
        X0_rank, int            : The rank of X0 (only used if X0 = 'qr-low-rank' or 'householder')
        rtol, float             : Stopping criterion, relative residual (r0/rk).
        seed, int|None          : Seed for random number generator (for initial X).
        verbose, bool           : Print out the results and progress.

        returns: Solution vector x = vec(X) (1D array).
        """
        # Initialize X (random)
        X = self.initial_X(seed, max_rank=X0_rank, X0=X0)[0]

        # Initialize gradient G and search direction D
        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        # Initial residual
        res0 = np.linalg.norm(G, 'fro')
        
        for i in range(1, max_iter + 1):
            # Step size
            HD = self.apply_H(D, w, lambda_)
            alpha = np.sum(G * G) / np.sum(D * HD)

            # Update X
            X = X + alpha * D
            
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
        
        self.niter = i
        return self.matrix_to_vec(X)
    
    def initial_X(
            self, seed: Optional[int], max_rank: int, X0: str
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
