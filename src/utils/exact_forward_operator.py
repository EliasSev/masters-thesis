"""
Construction of the exact forward operator K with corresponding weights W.
"""
import numpy as np
from numpy.typing import NDArray
from fenics import (
    FunctionSpace, DirichletBC, Constant, TrialFunction,
    TestFunction, dot, grad, dx, ds, assemble, Function
)


class ExactForwardOperator:
    def __init__(self, V_h: FunctionSpace):
        self.V_h = V_h
        self.bdofs = self._get_boundary_dofs()
        self.N = V_h.dim()
        self.N_b = len(self.bdofs)
        
        self.M_dx = self.assemble_M_dx()
        self.M_ds = self.assemble_M_ds()
        self.K = self.assemble_K()
    
    def _get_boundary_dofs(self) -> NDArray:
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.V_h, Constant(0.0), boundary)
        bc_dict = bc.get_boundary_values()
        return np.array(sorted(bc_dict.keys()), dtype=int)
    
    def assemble_S(self, sigma: float=1.0, k: float=1.0) -> NDArray:
        """Get the exact discrete PDE operator S."""
        print("Assembling S...")
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)

        sigma = Constant(sigma)
        k = Constant(k)
        a = sigma * dot(grad(u), grad(v)) * dx + k * u * v * dx  # LHS

        # Assemble A, M and S
        A = assemble(a).array()
        M = self.M_dx
        S = np.linalg.solve(A, M)  # S = A^{-1} @ M

        return S

    def assemble_T(self) -> NDArray:
        """Get the exact discrete trace operator T."""
        print("Assembling T...")
        T = np.zeros((self.N_b, self.N))
        for i, j in enumerate(self.bdofs):
            T[i, j] = 1.0
        return T

    def assemble_K(self) -> NDArray:
        """Get the exact discrete forward operator K"""
        S = self.assemble_S()
        T = self.assemble_T()
        return T @ S
    
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