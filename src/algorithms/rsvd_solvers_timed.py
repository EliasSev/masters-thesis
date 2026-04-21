"""
This file contains a modified version of rsvd_solvers.MatrixFreeRSVD 
which captures the computational time of the different stages of the algorithm.
"""
import numpy as np
from algorithms.rsvd_solvers import BaseSolver

from time import time
from typing import Optional
from numpy.typing import NDArray
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky

from fenics import FunctionSpace, set_log_level
set_log_level(30)


class MatrixFreeRSVDTimed(BaseSolver):
    def __init__(
            self,
            V_h: FunctionSpace,
            sigma: float = 1.0,
            c: float = 1.0,
            precompute: str = 'Cholesky'
        ) -> None:
        """
        Initialize a MatrixFreeRSVDTimed solver .
        
        Approximate the SVD of the forward operator K = TS, where T is
        the trace and S the PDE solve operator. The S operator solves the PDE:
            −∇ · σ∇u + cu = f, in Ω
            ∂u/∂n         = 0, on ∂Ω.

        V_h, FunctionSpace : A fenics FunctionSpace.
        sigma, float       : The diffusion coefficient.
        c, float           : The reaction coefficient.
        precompute, str    : Factorization method for M and M_ds.
        """
        t0 = time()
        super().__init__(V_h, sigma, c)

        if precompute == 'LU':
            self._solve_M_ds = splu(self.M_ds.tocsc()).solve

        elif precompute == 'Cholesky':
            self._solve_M_ds = cholesky(self.M_ds.tocsc()).solve_A

        else:
            raise ValueError(f"Unknown 'precompute': {precompute}")
        
        # Measure setup time
        self.setup_time = time() - t0
        self.times = None  # List of times of the 5 stages
    
    def solve(self,
            k: int,
            p: int = 5,
            distribution: str = 'standard',
            seed: Optional[int] = None,
            **kwargs
        ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Approximate the SVD of K using a matrix-free rSVD scheme.

        k, int            : Target rank.
        p, int            : Oversampling parameter.
        distribution, str : Distribution of the random test vectors psi.
        seed, int | None  : Seed for random number generator.
        """
        # Random number generator
        rng = np.random.default_rng(seed=seed)
        self.times = [self.setup_time]  # Step 1

        # Target rank k must be less than the rank of K
        assert k <= self.N_b, f"Target rank k={k} must be less than or equal to N_b={self.N_b}"
        l = min(k + p, self.N_b)

        # Range sketch
        t0 = time()
        Y = np.zeros((self.N_b, l))
        for i in range(l):
            psi_i = self._draw_random_vector(self.N, distribution, rng=rng, **kwargs)
            Y[:, i] = self.apply_K(psi_i)
        self.times.append(time() - t0)  # Step 2

        t0 = time()
        Q = np.linalg.qr(Y, mode='reduced')[0]
        self.times.append(time() - t0)  # Step 3
        
        # Projection onto basis
        t0 = time()
        B = np.zeros((l, self.N))
        for i in range(l):
            q_tilde = self._solve_M_ds(Q[:, i])
            b_tilde = self.apply_K_star(q_tilde).vector().get_local()
            B[i, :] = self.M @ b_tilde
        self.times.append(time() - t0)  # Step 4

        t0 = time()
        U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde
        self.times.append(time() - t0)  # Step 5

        # Truncate back to target rank
        self._U, self._S, self._Vt = U[:, :k], S[:k], Vt[:k, :]
        return self._U, self._S, self._Vt
