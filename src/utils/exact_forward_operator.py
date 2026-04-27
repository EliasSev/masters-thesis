"""
Construction of the exact forward operator K with corresponding weights W.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.sparse import csr_matrix, csc_matrix, spmatrix
from sksparse.cholmod import cholesky
from fenics import (
    FunctionSpace, DirichletBC, Constant, TrialFunction,
    TestFunction, dot, grad, dx, ds, assemble, Function,
    as_backend_type, set_log_level
)
set_log_level(30)


class ExactForwardOperator:
    def __init__(self, V_h: FunctionSpace, assemble_on_init: bool = True):
        self.V_h = V_h
        self.bdofs = self._get_boundary_dofs()
        self.N = V_h.dim()
        self.N_b = len(self.bdofs)
        
        self.M_dx = self.assemble_M_dx()
        self.M_ds = self.assemble_M_ds()

        if assemble_on_init:
            self.K = self.assemble_K()
        else:
            self.K = None
    
    def _get_boundary_dofs(self) -> NDArray:
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.V_h, Constant(0.0), boundary)
        bc_dict = bc.get_boundary_values()
        return np.array(sorted(bc_dict.keys()), dtype=int)
    
    def assemble_S(self, sigma: float=1.0, k: float=1.0) -> NDArray:
        """Get the exact discrete PDE operator S."""
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)

        sigma = Constant(sigma)
        k = Constant(k)
        a = sigma * dot(grad(u), grad(v)) * dx + k * u * v * dx  # LHS

        # Assemble A, M and S
        A = assemble(a).array()
        return solve(A, self.M_dx, assume_a="pos")  # S = A^{-1} @ M

    def assemble_T(self) -> NDArray:
        """Get the exact discrete trace operator T."""
        T = np.zeros((self.N_b, self.N))
        for i, j in enumerate(self.bdofs):
            T[i, j] = 1.0
        return T

    def assemble_K(self) -> NDArray:
        """Get the exact discrete forward operator K"""
        S = self.assemble_S()
        T = self.assemble_T()
        return T @ S

    def assemble_K_star(self, sigma: float = 1.0, k: float = 1.0) -> NDArray:
        """Get the exact discrete adjoint operator K^* = A^{-1} T^T M_ds, shape (N, N_b)."""
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        a = Constant(sigma) * dot(grad(u), grad(v)) * dx + Constant(k) * u * v * dx
        A = assemble(a).array()
        T = self.assemble_T()
        return solve(A, T.T @ self.M_ds, assume_a="pos")
    
    def assemble_M_dx(self, sigma: float=1.0, k: float=1.0) -> NDArray:
        """
        Get the mass matrix M_dx.
        """
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)

        sigma = Constant(sigma)
        k = Constant(k)

        m = u * v * dx
        return assemble(m).array()

    def assemble_M_ds(self):
        """
        Get the mass matrix M_ds.
        """
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        
        M_ds_full = assemble(u * v * ds)
        M_dense = M_ds_full.array()
        M_boundary = M_dense[np.ix_(self.bdofs, self.bdofs)]
        return M_boundary
    
    def get_weights(self, normalize: bool = True, tol: float = 1e-12) -> NDArray:
        """
        Get the regularization weights as a 1D array: w = diag(W).
        """
        _, s, Vt = np.linalg.svd(self.K, full_matrices=False)
        r = np.sum(s > (s[0] * tol))  # rank k of K

        Vr = Vt[:r, :].T 
        
        # Compute y_i^T * M_dx * y_i for all i
        # Mathematically: diag(R^T * M_dx * R)
        # Since R = Vr * Vr^T: diag(Vr * Vr^T * M_dx * Vr * Vr^T)
        # Let C = Vr^T * M_dx * Vr  (an r x r matrix)
        # Then w_sq = diag(Vr * C * Vr^T)
        C = Vr.T @ self.M_dx @ Vr
        w_sq = np.sum(Vr * (Vr @ C), axis=1)
        
        if normalize:
            volumes = np.array(self.M_dx.sum(axis=1)).flatten()
            w = np.sqrt(np.maximum(w_sq, 0)) / volumes
        else:
            w = w_sq
        
        return w
    

class ExactForwardOperatorFast:
    """
    Sparse Cholesky variant of `ExactForwardOperator`.

    Forms K = T A^{-1} M as a dense (N_b, N) array using a sparse Cholesky
    factorization of A and N_b sparse triangular solves (rather than N).

    Cost in 2D (sparse Cholesky with nested-dissection ordering):
        - Factorization of A: O(N^{3/2})
        - Forming K:          O(N_b * N log N) = O(N^{3/2} log N), since N_b ~ sqrt(N)

    The matrices M_dx and M_ds are stored as scipy CSR sparse matrices.
    """

    def __init__(
            self,
            V_h: FunctionSpace,
            sigma: float = 1.0,
            c: float = 1.0,
            assemble_on_init: bool = True,
        ):
        self.V_h = V_h
        self.sigma = sigma
        self.c = c
        self.bdofs = self._get_boundary_dofs()
        self.N = V_h.dim()
        self.N_b = len(self.bdofs)

        self.A = self._assemble_A()
        self.M_dx = self._assemble_M_dx()
        self.M_ds = self._assemble_M_ds()

        self.chol_A = cholesky(self.A.tocsc())

        self.K = self.assemble_K() if assemble_on_init else None

    def _get_boundary_dofs(self) -> NDArray:
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(self.V_h, Constant(0.0), boundary)
        bc_dict = bc.get_boundary_values()
        return np.array(sorted(bc_dict.keys()), dtype=int)

    def _assemble_sparse(self, form) -> csr_matrix:
        mat = as_backend_type(assemble(form)).mat()
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr))

    def _assemble_A(self) -> csr_matrix:
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        a = (Constant(self.sigma) * dot(grad(u), grad(v))
             + Constant(self.c) * u * v) * dx
        return self._assemble_sparse(a)

    def _assemble_M_dx(self) -> csr_matrix:
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        return self._assemble_sparse(u * v * dx)

    def _assemble_M_ds(self) -> csr_matrix:
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        M_full = self._assemble_sparse(u * v * ds)  # (N, N)
        return M_full[self.bdofs, :][:, self.bdofs]  # (N_b, N_b)

    def assemble_T(self) -> csr_matrix:
        """Trace operator T as sparse CSR, shape (N_b, N)."""
        rows = np.arange(self.N_b)
        return csr_matrix(
            (np.ones(self.N_b), (rows, self.bdofs)),
            shape=(self.N_b, self.N),
        )

    def assemble_K(self) -> NDArray:
        """
        K = T A^{-1} M, returned as a dense (N_b, N) array.

        Uses K = (M A^{-1} T^T)^T = (M Z)^T with Z = A^{-1} T^T (N_b solves).
        """
        T_T = csc_matrix(
            (np.ones(self.N_b), (self.bdofs, np.arange(self.N_b))),
            shape=(self.N, self.N_b),
        )
        Z = self.chol_A.solve_A(T_T.toarray())  # (N, N_b) dense
        return np.asarray((self.M_dx @ Z).T)    # (N_b, N) dense

    def assemble_K_star(self) -> NDArray:
        """
        K^* = A^{-1} T^T M_ds, returned as a dense (N, N_b) array.

        N_b solves of A against the sparse RHS T^T M_ds.
        """
        T_T = csc_matrix(
            (np.ones(self.N_b), (self.bdofs, np.arange(self.N_b))),
            shape=(self.N, self.N_b),
        )
        rhs = (T_T @ self.M_ds).toarray()     # (N, N_b) dense
        return np.asarray(self.chol_A.solve_A(rhs))


def solve_explicit(operator: ExactForwardOperator, w, y, lambda_):
    # Extract matrices
    K = operator.K
    M_dx = operator.M_dx
    M_ds = operator.M_ds
    W = np.diag(w)

    # Solve weighted Tikhonov problem
    LHS = K.T @ M_ds @ K + (lambda_**2) * (W @ M_dx @ W)
    RHS = K.T @ M_ds @ y
    x_hat = np.linalg.solve(LHS, RHS)

    f = Function(operator.V_h)
    f.vector()[:] = x_hat
    return f


def fast_get_weights(S: NDArray, Vt: NDArray, M: NDArray, tol: float = 1e-12):
        """
        Get the regularization weights as a 1D array: w = diag(W).

        S:  (N,) Singular values.
        Vt: (N, N) Right singular vectors.
        M:  (N, N) Mass matrix.
        """
        r = np.sum(S > (S[0] * tol))  # rank k of K
        Vr = Vt[:r, :].T 
        
        # Compute y_i^T * M_dx * y_i for all i
        # Mathematically: diag(R^T * M_dx * R)
        # Since R = Vr * Vr^T: diag(Vr * Vr^T * M_dx * Vr * Vr^T)
        # Let C = Vr^T * M_dx * Vr  (an r x r matrix)
        # Then w_sq = diag(Vr * C * Vr^T)
        C = Vr.T @ M @ Vr
        w_sq = np.sum(Vr * (Vr @ C), axis=1)
        volumes = np.array(M.sum(axis=1)).flatten()
        return np.sqrt(np.maximum(w_sq, 0)) / volumes