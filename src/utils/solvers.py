"""
Different solvers to find the inverse solution using the weighted Tikhonov problem.
"""
from numpy import diag, zeros
from numpy.typing import NDArray
from numpy.linalg import pinv, solve

from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import factorized, LinearOperator, cg

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


def fast_proj_solver_cg(
        Uk: NDArray, sk: NDArray, Vk: NDArray, M: NDArray,
        W_diag: NDArray, y: NDArray, lambda_: float, rtol: float = 1e-8
    ) -> NDArray:
    """
    Solves min_x || K^+ K x - K^+ y ||^2_M + lambda^2 || W x ||^2_M
    using conjugate gradient (CG).
    """
    N = Vk.shape[0]
    y_proj = Vk @ ((Uk.T @ y) / sk)   # y_proj = K^+ y = Vk @ Sigma^-1 @ Uk.T @ y
    rhs = Vk @ (Vk.T @ (M @ y_proj))  # Right-Hand Side (RHS): P @ M @ y_proj, P = V V^T
    
    # Action of the Hessian (LHS): H = P @ M @ P + lambda^2 W @ M @ W
    def lhs_action(x):
        # P @ M @ P @ x = V V^T M V V^T x
        term1 = Vk @ (Vk.T @ (M @ (Vk @ (Vk.T @ x))))
        # lambda^2 * W @ M @ W @ x
        term2 = (lambda_**2) * (W_diag * (M @ (W_diag * x)))
        return term1 + term2

    # Use CG to solve
    H_op = LinearOperator((N, N), matvec=lhs_action)
    x_opt, info = cg(H_op, rhs, rtol=rtol)
    
    if info > 0:
        print(f"Warning: CG did not converge. Info code: {info}")
    return x_opt


def fast_proj_solver(
        Uk: NDArray, sk: NDArray, Vk: NDArray, M: NDArray,
        W_diag: NDArray, y: NDArray, lambda_: float
    ) -> NDArray:
    """
    Solver for the projected Tikhonov problem.
    """
    N = Vk.shape[0]
    k = sk.shape[0]

    # 1. Setup the Sparse Part A
    W_mat = diags(W_diag)
    A = (lambda_**2) * (W_mat @ M @ W_mat)
    solve_A = factorized(csc_matrix(A))

    # 2. RHS Calculation
    y_proj = Vk @ ((Uk.T @ y) / sk)
    rhs = Vk @ (Vk.T @ (M @ y_proj))

    # 3. Woodbury components
    # Our low-rank part is V (V.T @ M @ V) V.T
    # Let S_M = V.T @ M @ V (a k x k dense matrix)
    S_M = Vk.T @ (M @ Vk)
    
    # 4. Compute Z = A^-1 @ V (N x k)
    # This is the 'heavy' part: k sparse back-solves
    Z = zeros((N, k))
    for i in range(k):
        Z[:, i] = solve_A(Vk[:, i])

    # 5. Capacitance Matrix C = inv(S_M) + V.T @ Z
    # If S_M is singular, we use pinv
    C = pinv(S_M) + Vk.T @ Z
    
    # 6. Final solve
    # x = A^-1 @ rhs - Z @ inv(C) @ (V.T @ A^-1 @ rhs)
    x0 = solve_A(rhs)
    rhs_small = Vk.T @ x0
    z = solve(C, rhs_small)
    
    return x0 - Z @ z
