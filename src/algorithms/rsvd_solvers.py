"""
This file contains:
    - `BaseSolver`: Parent class for PDE operator approximation solvers.
    - `MatrixFreeRSVD`: Improved implementation of the matrix-free rSVD scheme.
    - `MatrixFreeRSVDAdjoint`: Implementation of the matrix-free rSVD for K*.
"""
import numpy as np
from typing import Optional

from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix, spmatrix

from fenics import (
    FunctionSpace, Function, TrialFunction, TestFunction,
    Constant, DirichletBC, dot, grad, dx, ds, assemble,
    as_backend_type, LUSolver, Expression, interpolate, set_log_level
)
set_log_level(30)


class BaseSolver(ABC):
    def __init__(
            self,
            V_h: FunctionSpace,
            sigma: float = 1.0,
            c: float = 1.0,
            precompute: str = 'LU' 
        ) -> None:
        self.V_h = V_h
        self.sigma = sigma
        self.c = c

        self._U: Optional[NDArray] = None
        self._S: Optional[NDArray] = None
        self._Vt: Optional[NDArray] = None

        # Set up constants and constant matrices
        self.boundary_dofs = self._get_boundary_dofs()
        self.N = V_h.dim()
        self.N_b = len(self.boundary_dofs)
        self.n = int(np.sqrt(self.N))
        self.M = self._get_M()
        self.M_ds, self.M_ds_col = self._get_M_ds()

        # Set up a solver for S
        u, v = TrialFunction(V_h), TestFunction(V_h)
        self.A = assemble(Constant(sigma) * dot(grad(u), grad(v)) * dx + Constant(c) * u * v * dx)
        self.S_solver = LUSolver()  # or PETScKrylovSolver
        self.S_solver.set_operator(self.A)

        # Set up a solver for K*
        v, w = TrialFunction(V_h), TestFunction(V_h)
        self.A_star = assemble((dot(Constant(sigma) * grad(v), grad(w)) + Constant(c) * v * w) * dx)
        self.K_star_solver = LUSolver()  # or PETScKrylovSolver
        self.K_star_solver.set_operator(self.A_star)

        # Pre-allocate functions for `Apply_S`
        self._rhs_S = Function(self.V_h)
        self._sol_S = Function(self.V_h)

        # Pre-allocate functions for `Apply_K_star`
        self._rhs_Ks = Function(self.V_h)
        self._sol_Ks = Function(self.V_h)

        if precompute == 'LU':
            from scipy.sparse.linalg import splu
            self._solve_M    = splu(self.M.tocsc()).solve
            self._solve_M_ds = splu(self.M_ds.tocsc()).solve

        elif precompute == 'Cholesky':
            from sksparse.cholmod import cholesky
            self._solve_M    = cholesky(self.M.tocsc()).solve_A
            self._solve_M_ds = cholesky(self.M_ds.tocsc()).solve_A

        else:
            raise ValueError(f"Unknown 'precompute': {precompute}")

    @property 
    def U(self) -> NDArray:
        if self._U is None:
            raise ValueError("U is not computed yet.")
        return self._U
    
    @property 
    def S(self) -> NDArray:
        if self._S is None:
            raise ValueError("S is not computed yet.")
        return self._S
    
    @property 
    def Vt(self) -> NDArray:
        if self._Vt is None:
            raise ValueError("Vt is not computed yet.")
        return self._Vt
    
    @abstractmethod
    def solve(self) -> tuple[NDArray, NDArray, NDArray]:
        """Solver method to be implemented. Must return an array triplet; the approximate SVD"""
        pass
    
    def apply_K(self, x: NDArray) -> NDArray:
        """Apply the forward operator Kf = TSf = y."""
        u = self.apply_S(x)
        y = self.apply_T(u)
        return y
    
    def apply_S(self, x: NDArray) -> Function:
        """Apply the PDE operator Sf = u."""
        # ∫ f φᵢ dx = M x
        self._rhs_S.vector()[:] = self.M @ x
        self.S_solver.solve(self._sol_S.vector(), self._rhs_S.vector())
        return self._sol_S

    def apply_T(self, u: Function) -> NDArray:
        """Apply the trace operator Tu = y."""
        z = u.vector().get_local()
        y = z[self.boundary_dofs]
        return y
    
    def apply_K_star(self, y: NDArray) -> Function:
        """Apply the adjoint operator K*y = x"""
        # ∫∂Ω g φᵢ ds = M_ds_col y
        self._rhs_Ks.vector()[:] = self.M_ds_col @ y
        self.K_star_solver.solve(self._sol_Ks.vector(), self._rhs_Ks.vector())
        return self._sol_Ks
    
    def weights(self) -> NDArray:
        """
        Compute the Elvetun-Nielsen regularization weights:
            w_i = ∥V V^T (e_i)∥_M / sqrt(M_ii).
        """
        V = self.Vt.T
        C = self.Vt @ (self.M @ V)  # (k, N) @ ((N, k) @ (N, k))= (k×k)
        w_sq = np.sum(V * (V @ C), axis=1)
        
        volumes = np.array(self.M.sum(axis=1)).flatten()  # (N,)
        w = np.sqrt(np.maximum(w_sq, 0)) / volumes        # (N,)
        
        return np.array(w)
    
    def _get_M(self) -> spmatrix:
        """Get the mass matrix M as a scipy sparse CSR matrix."""
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        m = u * v * dx
        
        M = assemble(m)  # Assemble FEniCS matrix
        mat = as_backend_type(M).mat()  # Extract PETSc matrix
        
        # Construct sparse csr matrix
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr))

    def _get_M_ds(self) -> tuple[spmatrix, spmatrix]:
        """
        Assemble the boundary mass matrices M_ds and M_ds_col:
            M_ds:      (N_b, N_b)
            M_ds_col : (N, N_b)
        """
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)

        M_boundary = assemble(u * v * ds)
        mat = as_backend_type(M_boundary).mat()
        indptr, indices, data = mat.getValuesCSR()

        M_ds_full = csr_matrix((data, indices, indptr))                 # N × N
        M_ds = M_ds_full[self.boundary_dofs, :][:, self.boundary_dofs]  # N_b × N_b
        M_ds_col = M_ds_full[:, self.boundary_dofs]                     # N × N_b

        return M_ds, M_ds_col

    def _get_boundary_dofs(self) -> NDArray:
        """Given a FunctionSpace V_h, return the id's of the boundary nodes/dofs."""
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.V_h, Constant(0.0), boundary)
        bc_dict = bc.get_boundary_values()
        return np.array(sorted(bc_dict.keys()), dtype=int)
    
    def _draw_random_vector(
            self,
            d: int,
            distribution:str,
            rng: np.random.Generator,
            **kwargs
        ) -> NDArray:
        """
        Draw a random test vector psi of size d.
        """
        if distribution == 'standard':
            return rng.random(d)
        
        elif distribution == 'rademacher':
            return rng.choice([-0.5, 0.5], size=d, p=[0.5, 0.5])
        
        elif distribution == 'peaks':
            if d != self.N:
                raise ValueError("Can't use 'peaks' distribution; only implemented for d = self.N")
            
            x, y = rng.uniform(low=0, high=1, size=2)
            f_expr = Expression(
                "A*exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / (2*sigma*sigma))",
                degree=4, A=kwargs.get('A', 1.0), x0=x, y0=y, sigma=kwargs.get('sigma', 0.2)
            )
            f = interpolate(f_expr, self.V_h)
            return f.vector().get_local()

        else:
            raise ValueError(f"Unknown distribution: '{distribution}'")


class MatrixFreeRSVD(BaseSolver):
    def __init__(
            self,
            V_h: FunctionSpace,
            sigma: float = 1.0,
            c: float = 1.0,
            precompute: str = 'LU' 
        ) -> None:
        """
        Initialize a MatrixFreeRSVD solver.
        
        Approximate the SVD of the forward operator K = TS, where T is
        the trace and S the PDE solve operator. The S operator solves the PDE:
            −∇ · σ∇u + cu = f, in Ω
            ∂u/∂n         = 0, on ∂Ω.

        V_h, FunctionSpace : A fenics FunctionSpace.
        sigma, float       : The diffusion coefficient.
        c, float           : The reaction coefficient.
        """
        super().__init__(V_h, sigma, c, precompute)
    
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

        # Target rank k must be less than dim(Nul(V_h)) = N_b
        assert k <= self.N_b, f"Target rank k={k} must be less than or equal to N_b={self.N_b}"
        l = min(k + p, self.N_b)

        # Range sketch
        Y = np.zeros((self.N_b, l))
        for i in range(l):
            psi_i = self._draw_random_vector(self.N, distribution, rng=rng, **kwargs)
            Y[:, i] = self.apply_K(psi_i)

        Q = np.linalg.qr(Y, mode='reduced')[0]
        
        # Projection onto basis
        B = np.zeros((l, self.N))
        for i in range(l):
            q_tilde = self._solve_M_ds(Q[:, i])
            b_tilde = self.apply_K_star(q_tilde).vector().get_local()
            B[i, :] = self.M @ b_tilde

        U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde

        # Truncate back to target rank
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]
        self._U, self._S, self._Vt = U, S, Vt
        return U, S, Vt


class MatrixFreeRSVDAdjoint(BaseSolver):
    def __init__(
            self,
            V_h: FunctionSpace,
            sigma: float = 1.0,
            c: float = 1.0,
            precompute: str = 'LU' 
        ) -> None:
        """
        Initialize a MatrixFreeRSVDAdjoint solver.
        
        Approximate the SVD of the adjoint operator K* = S*T*, where T is
        the trace and S the PDE solve operator. The S operator solves the PDE:
            −∇ · σ∇u + cu = f, in Ω
            ∂u/∂n         = 0, on ∂Ω.

        V_h, FunctionSpace : A fenics FunctionSpace.
        sigma, float       : The diffusion coefficient.
        c, float           : The reaction coefficient.
        """
        super().__init__(V_h, sigma, c, precompute)
    
    def solve(self,
            k: int,
            p: int = 5,
            distribution: str = 'standard',
            seed: Optional[int] = None,
            **kwargs
        ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Approximate the SVD of K* using a matrix-free rSVD scheme.

        k, int            : Target rank.
        p, int            : Oversampling parameter.
        distribution, str : Distribution of the random test vectors psi.
        seed, int | None  : Seed for random number generator.
        """
        # Random number generator
        rng = np.random.default_rng(seed=seed)

        # Target rank k must be less than the null-space dim N_b
        assert k <= self.N_b, f"Target rank k={k} must be less than or equal to N_b={self.N_b}"
        l = min(k + p, self.N_b)

        # Range sketch
        Y = np.zeros((self.N, l))
        for i in range(l):
            psi_i = self._draw_random_vector(self.N_b, distribution, rng=rng, **kwargs)
            Y[:, i] = self.apply_K_star(psi_i).vector().get_local()

        Q = np.linalg.qr(Y, mode='reduced')[0]
        
        # Projection onto basis
        B = np.zeros((l, self.N_b))
        for i in range(l):
            q_tilde = self._solve_M(Q[:, i])
            b_tilde = self.apply_K(q_tilde)
            B[i, :] = self.M_ds @ b_tilde

        U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde

        # Truncate back to target rank
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]
        self._U, self._S, self._Vt = U, S, Vt
        return U, S, Vt
