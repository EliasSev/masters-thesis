"""
Different solvers to find the inverse solution using the weighted Tikhonov problem.
"""
from numpy import diag, zeros
from numpy.typing import NDArray
from numpy.linalg import pinv, solve

from scipy.sparse import diags
from scipy.sparse.linalg import factorized

from fenics import Function
from algorithms.matrix_free_rsvd import MatrixFreeRSVD


def fast_rsvd_solver(
        y: NDArray, w: NDArray, lam: float, rsvd: MatrixFreeRSVD
    ) -> Function:
    """
    A wrapper around `fast_solver`, which takes in an rSVD object
    and calls `fast_solver`, and returns the solution as a fenics Function.
    """
    f_hat = Function(rsvd.V_h)
    f_hat.vector()[:] = fast_svd_solver(
        y = y,
        U = rsvd.Uk,
        s = rsvd.Sk,
        V = rsvd.VkT.T,
        M_partial = rsvd.M_ds,
        M = rsvd.M_dx,
        w = w,
        lam = lam
    )
    return f_hat


def fast_svd_solver(
        y: NDArray, U: NDArray, s: NDArray, V: NDArray,
        M_partial: NDArray, M: NDArray, w: NDArray, lam: float
    ) -> NDArray:
    """
    Solve H x = K.T @ M_partial @ y, i.e. minimize
        Phi(x) = 0.5 * ||Kx - y||^2 + 0.5 * lam**2 * ||Wx||

    y          : (N_b, ) Boundary observation.
    U          : (N_b, k) Left singular values of K.
    s          : (k, ) Singular values of K.
    V          : (N, k) Right singular values of K.
    M_partial  : (N_b, N_b) Boundary mass matrix.
    M          : (N, N) Mass matrix.
    w          : (N,) Regularization weights.
    lam        : Regularization parameter
    """
    N = V.shape[0]
    k = s.shape[0]

    # Construct the Right-Hand Side (RHS)
    RHS = V @ (s * (U.T @ (M_partial @ y)))  # Size (N,)

    # Setup Sparse Part A
    W_mat = diags(w)
    A = (lam ** 2) * (W_mat @ M @ W_mat)
    solve_A = factorized(A.tocsc())

    # Setup Low-Rank Update components
    # H = A + V @ S @ V.T
    # S = Sigma @ U.T @ M_partial @ U @ Sigma
    S_mid = U.T @ (M_partial @ U)
    S = diag(s) @ S_mid @ diag(s)
    S_inv = pinv(S)  # or just np.linalg.inv

    # Woodbury/SMW Solve
    # Compute Z = A^-1 @ V (N x k)
    Z = zeros((N, k))
    for i in range(k):
        Z[:, i] = solve_A(V[:, i])

    C = S_inv + V.T @ Z
    x0 = solve_A(RHS)  # x0 = A^-1 @ RHS
    
    rhs_small = V.T @ x0
    z = solve(C, rhs_small)
    x_opt = x0 - Z @ z
    return x_opt
