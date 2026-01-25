import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from fenics import (
    FunctionSpace, Function, TrialFunction, TestFunction,
    Constant, DirichletBC, dot, grad, dx, ds, solve, assemble,
    as_backend_type
)


class MatrixFreeRSVD:
    def __init__(self, V_h: FunctionSpace, sigma: float = 1.0, k: float = 1.0):
        self.V_h = V_h
        self.sigma = sigma
        self.k = k

        # Setup constants and constant matrices
        self.boundary_dofs = self.get_boundary_dofs()
        self.N = V_h.dim()
        self.N_b = len(self.boundary_dofs)
        self.M_dx = self.get_M_dx()
        self.M_ds = self.get_M_ds()

        # Variational form for `apply_S`
        u = TrialFunction(V_h)
        v = TestFunction(V_h)
        self.A = Constant(sigma) * dot(grad(u), grad(v)) * dx + Constant(k) * u * v * dx

        # Variational form for `apply_K_star`
        v = TrialFunction(V_h)
        w = TestFunction(V_h)
        self.A_star = (dot(Constant(sigma) * grad(v), grad(w)) + Constant(k) * v * w) * dx
    
    def mf_rsvd(self, k: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Implementation of the Discrete Operator rSVD algorithms, which approximates
        the discrete operator K: f -> u through random sampling of the operator.
        """
        # Step 1
        Y = np.zeros((self.N_b, k))
        for i in range(k):
            psi_i = np.random.randn(self.N)
            y_i = self.apply_K(psi_i)
            Y[:, i] = y_i

        # Step 2
        Q, _ = np.linalg.qr(Y, mode='reduced')

        # Step 3-4
        B = np.zeros((k, self.N))
        for i in range(k):
            q_i = Q[:, i].copy()

            # Get the "un-weight" q_i to counteract the M_ds inside apply_K_adj
            q_unweighted = spsolve(self.M_ds, q_i) 
            b_mod = self.apply_K_star(q_unweighted).vector().get_local()
            B[i, :] = b_mod.T @ self.M_dx

        # Step 5-6
        U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde
        return U, S, Vt
    
    def apply_K(self, x: NDArray) -> NDArray:
        """
        Forward operator K = (T)(S).
        """
        u = self.apply_S(x)
        y = self.apply_T(u)
        return y
    
    def apply_S(self, x: NDArray) -> Function:
        """
        Apply the PDE operator S: x -> z.
        """
        v = TestFunction(self.V_h)

        f = Function(self.V_h)
        f.vector()[:] = x  # x is the nodal coefficient vector
        L = f * v * dx

        u_sol = Function(self.V_h)
        solve(self.A == L, u_sol)
        return u_sol

    def apply_T(self, u: Function) -> NDArray:
        """
        Apply the trace operator T: z -> x.
        """
        z = u.vector().get_local()
        y = z[self.boundary_dofs]
        return y
    
    def apply_K_star(self, y: NDArray) -> Function:
        """
        Implementation of K*: y -> x.
        """
        # Define the function g  (is this correct?)
        y_filled = np.zeros(self.N)
        y_filled[self.boundary_dofs] = y
        g = Function(self.V_h)
        g.vector()[:] = y_filled
        
        w = TestFunction(self.V_h)
        L_adj = g * w * ds  # Surface integral

        f = Function(self.V_h)
        solve(self.A_star == L_adj, f)
        return f

    def get_M_dx(self) -> csr_matrix:
        """
        Get the mass matrix M_dx as a scipy sparse CSR matrix.
        """
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        m = u * v * dx
        
        M = assemble(m)  # Assemble FEniCS matrix
        mat = as_backend_type(M).mat()  # Extract PETSc matrix
        
        # Construct sparse csr matrix
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr))

    def get_M_ds(self) -> csr_matrix:
        """
        Get the boundary mass matrix M_ds as a scipy sparse CSR matrix.
        """
        u = TrialFunction(self.V_h)
        v = TestFunction(self.V_h)
        m_ds = u * v * ds 
        
        M_boundary = assemble(m_ds)
        mat = as_backend_type(M_boundary).mat()
        indptr, indices, data = mat.getValuesCSR()

        M_ds = csr_matrix((data, indices, indptr))
        return M_ds[self.boundary_dofs, :][:, self.boundary_dofs]

    def get_boundary_dofs(self) -> NDArray:
        """
        Given a FunctionSpace V_h, return the id's of the boundary nodes/dofs.
        """
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(self.V_h, Constant(0.0), boundary)
        bc_dict = bc.get_boundary_values()
        return np.array(sorted(bc_dict.keys()), dtype=int)
