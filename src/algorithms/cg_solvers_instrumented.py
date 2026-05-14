"""
Instrumented CG / low-rank solvers used for diagnostics, not production runs.

DynamicalLowRankPCGConjugacy: subclass of DynamicalLowRankPCG that, on top of
the usual residual/error history, records the H-conjugacy loss

    ell(i) = max_{j < i} | C_ij / sqrt(C_ii * C_jj) |,

with C_ij = <D^(i), H D^(j)>_F. Stored on .conjugacy_loss after solve().

DynamicalLowRankCGShadow: subclass of DynamicalLowRankCG that tracks the
"shadow" iterate

    X_tilde^(i) = X^(0) + sum_{j < i} alpha_j D^(j),

i.e. the trajectory plain CG would follow with the same (alpha_j, D^(j))
sequence but no KLS / truncation. The Frobenius distance between X_tilde^(i)
and the manifold iterate X^(i) is recorded each iteration, along with the
tangent fraction ||P_{X^(i)} D^(i)||_F / ||D^(i)||_F that quantifies how much
of the search direction the manifold can use at each step.

Memory: keeps every H D^(j) as a dense (n, n) array, so this is only practical
on small meshes (n <= ~32 in TestProblemsSetup terms).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from algorithms.cg_solvers import (
    DynamicalLowRankCG,
    DynamicalLowRankPCG,
    frobenius2,
    inner_F,
)
from utils.utils import progress_bar


class DynamicalLowRankPCGConjugacy(DynamicalLowRankPCG):
    """DLR-PCG instrumented with H-conjugacy tracking. Drop-in for diagnostics."""

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            max_rank: int = 5,
            *,
            preconditioner: str = 'ic',
            truncate_tol: float = 0.01,
            X0: str = 'qr',
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: float = 0.0,
            seed: Optional[int] = None,
            verbose: bool = True,
            restart_every: Optional[int] = None,
        ) -> NDArray:
        """
        Same signature as DynamicalLowRankPCG.solve, plus:
          - self.conjugacy_loss: list of ell(i), one per iteration
          - self.C_diag:         list of C_ii = <D^(i), H D^(i)>_F
          - self.C_rows:         list of arrays, row i is [C_{i,j}]_{j<i}
          - self.restart_iters:  iteration indices where the CG history was reset

        After a restart the H-conjugacy history is cleared, since D is reset to
        -Z and is unrelated to the prior search directions.
        """
        lambda_ = lambda_**2  # match thesis convention used in parent
        self.error, self.residual = [1.0], [1.0]

        # Conjugacy bookkeeping
        self.conjugacy_loss = []
        self.C_diag = []
        self.C_rows = []
        self.restart_iters = []
        HD_history = []  # local: not exposed (large)

        X, Ux, Sx, Vx = self.initial_X(seed, max_rank, X0)
        P_inv = self.get_preconditioner(w, lambda_, preconditioner)

        G = self.gradient(X, y, w, lambda_)
        Z = self.apply_P_inv(G, P_inv)
        D = -Z.copy()

        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        for i in range(1, max_iter + 1):
            HD = self.apply_H(D, w, lambda_)
            denom = inner_F(D, HD)            # = C_ii
            alpha = inner_F(G, Z) / denom

            # H-conjugacy row for D^(i) against past D^(j)
            C_ii = denom
            if HD_history:
                C_row = np.fromiter(
                    (inner_F(D, HD_j) for HD_j in HD_history),
                    dtype=float, count=len(HD_history),
                )
                C_diag_arr = np.asarray(self.C_diag)
                cos_abs = np.abs(C_row) / np.sqrt(C_ii * C_diag_arr)
                ell = float(cos_abs.max())
            else:
                C_row = np.empty(0)
                ell = 0.0

            self.conjugacy_loss.append(ell)
            self.C_diag.append(C_ii)
            self.C_rows.append(C_row)
            HD_history.append(HD.copy())

            # W-step
            W_star = (Ux @ Sx) + alpha * (D @ Vx)
            U_hat, _ = np.linalg.qr(np.hstack([Ux, W_star]))

            # L-step
            L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new + alpha * (U_hat.T @ D @ V_hat)

            # Truncate back to low-rank
            Ux, Sx, Vx = self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)

            # Update G, Z, D (with restart handling)
            if restart_every is not None and i % restart_every == 0:
                X = Ux @ Sx @ Vx.T
                G = self.gradient(X, y, w, lambda_)
                Z = self.apply_P_inv(G, P_inv)
                D = -Z.copy()
                # Reset conjugacy history: the new D is unrelated to previous D's.
                HD_history.clear()
                self.C_diag.clear()
                self.restart_iters.append(i)
            else:
                denom_PR = inner_F(G, Z)
                G = G + alpha * HD
                Z = self.apply_P_inv(G, P_inv)
                beta = inner_F(G, Z) / denom_PR
                D = -Z + beta * D

            # Relative residual / error
            res = np.sqrt(frobenius2(G))
            self.residual.append(res / res0)
            err = np.sqrt(frobenius2(Ux @ Sx @ Vx.T - self.X_true))
            self.error.append(err / err0)

            if self.residual[-1] < rtol:
                if verbose:
                    print(f"Converged at iter {i} [rel_res={self.residual[-1]:.3}]")
                break
            if self.error[-1] < etol:
                if verbose:
                    print(f"Converged at iter {i} [etol criteria: rel_err={self.error[-1]:.3}]")
                break

            if verbose and ((i % 10 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def conjugacy_matrix(self) -> NDArray:
        """
        Assemble the symmetric (K, K) matrix |C_tilde_ij| of normalised
        H-cosines from the most recent solve. K is the number of iterations
        in the *current* CG segment (after the last restart, if any).
        """
        K = len(self.C_diag)
        C = np.eye(K)
        diag = np.asarray(self.C_diag)
        for i, row in enumerate(self.C_rows[-K:]):
            if row.size == 0:
                continue
            cos = np.abs(row) / np.sqrt(diag[i] * diag[: row.size])
            C[i, : row.size] = cos
            C[: row.size, i] = cos
        return C


class DynamicalLowRankCGShadow(DynamicalLowRankCG):
    """DLR-CG instrumented with shadow-trajectory tracking. Drop-in for diagnostics."""

    def solve(
            self,
            y: NDArray,
            w: NDArray,
            lambda_: float = 1e-4,
            *,
            X0: str = 'qr',
            max_rank: int = 5,
            max_iter: int = 250,
            rtol: float = 1e-8,
            etol: Optional[float] = None,
            seed: Optional[int] = None,
            verbose: bool = True,
            truncate_tol: float = 0.01,
            restart_every: Optional[int] = None,
        ) -> NDArray:
        """
        Same signature as DynamicalLowRankCG.solve, plus:
          - self.shadow_distance:     ||X_tilde^(i) - X^(i)||_F per iteration
          - self.shadow_distance_rel: ||X_tilde^(i) - X^(i)||_F / ||X_tilde^(i)||_F
          - self.shadow_norm:         ||X_tilde^(i)||_F per iteration
          - self.tangent_fraction:    ||P_{X^(i)} D^(i)||_F / ||D^(i)||_F
          - self.restart_iters:       iterations where the shadow was re-anchored

        On restart, the gradient recurrence is re-anchored to X^(i), so the
        shadow is reset to the current manifold iterate.
        """
        lambda_ = lambda_**2
        self.residual, self.error = [1.0], [1.0]

        # Shadow bookkeeping (i = 0: shadow == manifold iterate by construction)
        self.shadow_distance = [0.0]
        self.shadow_distance_rel = [0.0]
        self.shadow_norm = []
        self.tangent_fraction = []
        self.restart_iters = []

        X, Ux, Sx, Vx = self.initial_X(seed, max_rank=max_rank, X0=X0)
        X_shadow = X.copy()
        self.shadow_norm.append(float(np.sqrt(frobenius2(X_shadow))))

        G = self.gradient(X, y, w, lambda_)
        D = -G.copy()

        # Initial tangent fraction at (X^(0), D^(0) = -G^(0))
        self.tangent_fraction.append(self._tangent_fraction(D, Ux, Vx))

        res0 = np.sqrt(frobenius2(G))
        err0 = np.sqrt(frobenius2(X - self.X_true))

        for i in range(1, max_iter + 1):
            HD = self.apply_H(D, w, lambda_)
            alpha = frobenius2(G) / inner_F(D, HD)

            # Advance shadow before the manifold step (same alpha, D).
            X_shadow = X_shadow + alpha * D

            # W-step
            W_star = (Ux @ Sx) + alpha * (D @ Vx)
            U_hat, _ = np.linalg.qr(np.hstack([Ux, W_star]))

            # L-step
            L_star = (Vx @ Sx.T) + alpha * (D.T @ Ux)
            V_hat, _ = np.linalg.qr(np.hstack([Vx, L_star]))

            # S-step
            S_new = (U_hat.T @ Ux) @ Sx @ (Vx.T @ V_hat)
            S_new = S_new + alpha * (U_hat.T @ D @ V_hat)

            # Truncate back to low-rank
            Ux, Sx, Vx = self.truncate(U_hat, S_new, V_hat, truncate_tol, max_rank)

            X_curr = Ux @ Sx @ Vx.T

            # Update G, D (with restart handling)
            if restart_every is not None and i % restart_every == 0:
                G = self.gradient(X_curr, y, w, lambda_)
                D = -G.copy()
                # Restart re-anchors the recurrence to X^(i): reset shadow.
                X_shadow = X_curr.copy()
                self.restart_iters.append(i)
            else:
                denom = frobenius2(G)
                G = G + alpha * HD
                beta = frobenius2(G) / denom
                D = -G + beta * D

            shadow_norm = float(np.sqrt(frobenius2(X_shadow)))
            dist = float(np.sqrt(frobenius2(X_shadow - X_curr)))
            self.shadow_norm.append(shadow_norm)
            self.shadow_distance.append(dist)
            self.shadow_distance_rel.append(
                dist / shadow_norm if shadow_norm > 0 else 0.0
            )
            self.tangent_fraction.append(self._tangent_fraction(D, Ux, Vx))

            res = np.sqrt(frobenius2(G))
            self.residual.append(res / res0)
            err = np.sqrt(frobenius2(X_curr - self.X_true))
            self.error.append(err / err0)

            if self.residual[-1] < rtol:
                if verbose:
                    print(f"Converged at iter {i} [rtol criteria: rel_res={self.residual[-1]:.3}]")
                break
            if etol is not None and self.error[-1] < etol:
                if verbose:
                    print(f"Converged at iter {i} [etol criteria: rel_err={self.error[-1]:.3}]")
                break

            if verbose and ((i % 100 == 0) or (i == max_iter)):
                progress_bar(i, max_iter)

        self.niter = i
        return self.matrix_to_vec(Ux @ Sx @ Vx.T)

    def _tangent_fraction(self, D: NDArray, Ux: NDArray, Vx: NDArray) -> float:
        """||P_{X} D||_F / ||D||_F, where P_{X} is the tangent projector at
        X = Ux Sx Vx^T (independent of Sx)."""
        D_norm2 = frobenius2(D)
        if D_norm2 == 0.0:
            return 0.0
        xi, *_ = self.project_tangent(D, Ux, Vx)
        return float(np.sqrt(frobenius2(xi) / D_norm2))
