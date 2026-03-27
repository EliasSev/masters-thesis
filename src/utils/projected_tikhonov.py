"""
Utilities related to the projected Tikhonov problem.
"""
import numpy as np
from typing import Optional
from utils.solvers import fast_proj_solver_cg


def get_epsilon(y: np.ndarray, relative: float, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    epsilon = rng.standard_normal(y.shape)
    epsilon /= np.linalg.norm(epsilon)
    return epsilon * relative * np.linalg.norm(y)


def discrepancy_principle(y, U, S, V, M, w, norm_epsilon, nu=5.0, num_lam=10, rtol=1e-8):
    target = nu * norm_epsilon

    for lam in np.logspace(-4, 1, num_lam):
        x_lam = fast_proj_solver_cg(U, S, V, M, w, y, lambda_=lam, rtol=rtol)
        Kx_lam = U @ (S * (V.T @ x_lam))
        if np.linalg.norm(Kx_lam - y) > target:
            return lam
    else:
        print("Warning: did not find lambda. Setting lambda=1e-2")
        return 1e-2
