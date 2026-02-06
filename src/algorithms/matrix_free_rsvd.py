import numpy as np
from time import time
from typing import Optional
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.linalg import spsolve, LinearOperator, cg
from fenics import (
    FunctionSpace, Function, TrialFunction, TestFunction,
    Constant, DirichletBC, dot, grad, dx, ds, assemble,
    as_backend_type, LUSolver, Expression, interpolate, set_log_level
)
set_log_level(30)


class MatrixFreeRSVD:
    def __init__(self, V_h: FunctionSpace, sigma: float = 1.0, k: float = 1.0):
        self.V_h = V_h
        self.sigma = sigma
        self.k = k
        self._Uk: Optional[NDArray] = None
        self._Sk: Optional[NDArray] = None
        self._VkT: Optional[NDArray] = None
        self.times = None

        # Setup constants and constant matrices
        self.boundary_dofs = self.get_boundary_dofs()
        self.N = V_h.dim()
        self.N_b = len(self.boundary_dofs)
        self.M_dx = self.get_M_dx()
        self.M_ds = self.get_M_ds()

        # Variational form for `apply_S`
        u = TrialFunction(V_h)
        v = TestFunction(V_h)
        self.A = assemble(Constant(sigma) * dot(grad(u), grad(v)) * dx + Constant(k) * u * v * dx)
        self.S_solver = LUSolver()  # or PETScKrylovSolver
        self.S_solver.set_operator(self.A)

        # Variational form for `apply_K_star`
        v = TrialFunction(V_h)
        w = TestFunction(V_h)
        self.A_star = assemble((dot(Constant(sigma) * grad(v), grad(w)) + Constant(k) * v * w) * dx)
        self.K_star_solver = LUSolver()  # or PETScKrylovSolver
        self.K_star_solver.set_operator(self.A_star)

    @property 
    def Uk(self) -> NDArray:
        if self._Uk is None:
            raise ValueError("Uk is not computed yet.")
        return self._Uk
    
    @property 
    def Sk(self) -> NDArray:
        if self._Sk is None:
            raise ValueError("Sk is not computed yet.")
        return self._Sk
    
    @property 
    def VkT(self) -> NDArray:
        if self._VkT is None:
            raise ValueError("VkT is not computed yet.")
        return self._VkT
    
    def mf_rsvd(self,
            k: int, distribution: str = 'standard', seed: Optional[int] = None
        ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Implementation of the Discrete Operator rSVD algorithms, which approximates
        the discrete operator K: f -> u through random sampling of the operator.
        """
        # Random number generator
        rng = np.random.default_rng(seed=seed)

        self.times = []
        # Step 1
        t0 = time()
        Y = np.zeros((self.N_b, k))
        for i in range(k):
            psi_i = self.draw_random_vector(distribution, rng=rng)
            y_i = self.apply_K(psi_i)
            Y[:, i] = y_i
        self.times.append(time() - t0)

        # Step 2
        t0 = time()
        Q, _ = np.linalg.qr(Y, mode='reduced')
        self.times.append(time() - t0)

        # Step 3-4
        t0 = time()
        B = np.zeros((k, self.N))
        for i in range(k):
            q_i = Q[:, i].copy()

            # Get the "un-weight" q_i to counteract the M_ds inside apply_K_adj
            q_unweighted = spsolve(self.M_ds, q_i) 
            b_mod = self.apply_K_star(q_unweighted).vector().get_local()
            B[i, :] = b_mod.T @ self.M_dx
        self.times.append(time() - t0)

        # Step 5-6
        t0 = time()
        U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde
        self.times.append(time() - t0)
        
        self._Uk, self._Sk, self._VkT = U, S, Vt
        return U, S, Vt
    
    def draw_random_vector(self, distribution: str, rng: np.random.Generator, **kwargs) -> NDArray:
        if distribution == 'standard':
            return rng.random(self.N)
        
        elif distribution == 'rademacher':
            return rng.choice([-0.5, 0.5], size=self.N, p=[0.5, 0.5])
        
        elif distribution == 'peaks':
            x, y = rng.uniform(low=0, high=1, size=2)
            f_expr = Expression(
                "A*exp(-((x[0]-x0)*(x[0]-x0) + (x[1]-y0)*(x[1]-y0)) / (2*sigma*sigma))",
                degree=4, A=kwargs.get('A', 1.0), x0=x, y0=y, sigma=kwargs.get('sigma', 0.2)
            )
            f = interpolate(f_expr, self.V_h)
            return f.vector().get_local()

        else:
            raise ValueError(f"Unknown distribution: '{distribution}'")
    
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

        # TODO: make f attribute, update its coefficients
        # then L can also be kept as an attribute, dependent of f 
        f = Function(self.V_h)
        f.vector()[:] = x  # x is the nodal coefficient vector
        L = f * v * dx
        rhs = assemble(L)

        u_sol = Function(self.V_h)
        self.S_solver.solve(u_sol.vector(), rhs)
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
        L_star = g * w * ds  # Surface integral
        rhs = assemble(L_star)

        v_sol = Function(self.V_h)
        self.K_star_solver.solve(v_sol.vector(), rhs)
        return v_sol

    def get_M_dx(self) -> spmatrix:
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

    def get_M_ds(self) -> spmatrix:
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


def get_approximate_W(Vk: NDArray, M_dx: spmatrix) -> NDArray:
    C = Vk.T @ M_dx @ Vk
    w_sq = np.sum(Vk * (Vk @ C), axis=1)
    
    volumes = np.array(M_dx.sum(axis=1)).flatten()
    w = np.sqrt(np.maximum(w_sq, 0)) / volumes
    
    return np.array(w)


def tikhonov_solver(rsvd: MatrixFreeRSVD, W_diag: NDArray, y: NDArray, lambda_: float) -> NDArray:
    """
    Solves (K^T M_ds K + lambda^2 W M_dx W) f = K^T M_ds y
    using the rank-k SVD components: K = Uk @ diag(sk) @ Vk.T
    """
    Uk = rsvd.Uk
    sk = rsvd.Sk
    Vk = rsvd.VkT.T
    M_ds = rsvd.M_ds
    M_dx = rsvd.M_dx

    N = Vk.shape[0]
    W = W_diag # Assuming this is a 1D array of weights
    
    # Pre-compute the Right Hand Side (RHS)
    # K^T * M_ds * y
    rhs = Vk @ (sk * (Uk.T @ (M_ds @ y)))

    # Define the Matrix-Vector Product (Action of the LHS)
    def lhs_action(f):
        # 1. Term: K^T @ M_ds @ K @ f
        # Forward: K @ f
        Kf = Uk @ (sk * (Vk.T @ f))
        # Adjoint: K^T @ (M_ds @ Kf)
        term1 = Vk @ (sk * (Uk.T @ (M_ds @ Kf)))
        
        # 2. Term: lambda^2 @ W @ M_dx @ W @ f
        # Since W is diagonal, W @ f is element-wise multiplication
        Wf = W * f
        term2 = (lambda_**2) * (W * (M_dx @ Wf))
        
        return term1 + term2

    # Wrap as a LinearOperator
    A_op = LinearOperator((N, N), matvec=lhs_action)

    # Solve using Conjugate Gradient
    # tol: stop when residual is small enough
    f_hat, info = cg(A_op, rhs, rtol=1e-8)
    
    if info > 0:
        print("Warning: CG did not converge")
        
    return f_hat
