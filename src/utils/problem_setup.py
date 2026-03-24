"""
A class to set up the three different test problems (I, II, III) used in the thesis.
"""
import numpy as np

from typing import Any
from numpy.typing import NDArray
from fenics import Function, FunctionSpace

from utils.mesh_utils import get_square_mesh, get_L_mesh, get_square_f
from algorithms.matrix_free_rsvd import MatrixFreeRSVD, get_approximate_W


class TestProblemsSetup:
    def __init__(self, n: int) -> None:
        """
        Initialize a TestProblemsSetup instance.
        
        n, int : The resolution of the mesh (approx n x n nodes).
        """

        # Parameters for the setup of the 3 test problems
        self.problem_params = {
            'I': {
                'mesh': get_square_mesh,
                'n': n,
                'width': 0.15,
                'height': 0.15,
                'x0': [0.2],
                'y0': [0.2]
            },
            'II': {
                'mesh': get_square_mesh,
                'n': n,
                'width': 0.15,
                'height': 0.15,
                'x0': [0.1, 0.75, 0.15],
                'y0': [0.1, 0.75, 0.70]
            },
            'III': {
                'mesh': get_L_mesh,
                'n': n,
                'width': 0.25,
                'height': 0.25,
                'x0': [0.2, 1.55],
                'y0': [0.2, 0.55]
            }
        }

    def get_test_problems(self, verbose: bool = False, k: int = 50) -> dict[str, dict]:
        """Set up all three problems (I, II, II)"""
        problems =  {}
        for key, params in self.problem_params.items():
            if verbose: print(f"Setting up problem {key}: ", end='')
            pb = self.problem_setup(params, k=k)

            if verbose: print(f" N_b={pb['rsvd'].N_b}, N={pb['rsvd'].N} (done)")
            problems[key] = pb

        return problems
    
    def problem_setup(self, params: dict[str, Any], k: int = 50) -> dict[str, Any]:
        """Set up a single test problem."""
        # Function space setup
        mesh = params['mesh'](params['n'])
        V_h = FunctionSpace(mesh, 'CG', 1)
        rsvd = MatrixFreeRSVD(V_h)
        rsvd.mf_rsvd(k=k)
        w = get_approximate_W(Vk=rsvd.VkT.T, M_dx=rsvd.M_dx)

        # Source setup
        f, x = self.get_source(
            V_h = V_h,
            x0_list = params['x0'],
            y0_list = params['y0'], 
            width = params['width'],
            height = params['height']
        )
        y = rsvd.apply_K(x)
        return {'V_h': V_h, 'rsvd': rsvd, 'w': w, 'f': f, 'x': x, 'y': y}
    
    def get_source(
            self, V_h: FunctionSpace, x0_list: list, y0_list: list, width: float, height: float
        ) -> tuple[Function, NDArray]:
        """Construct a source function f and its coefficient vector x."""
        x = np.zeros(V_h.dim())
        for x0, y0 in zip(x0_list, y0_list):
            f = get_square_f(V_h, x0=x0, y0=y0, w=width, h=height)
            x += f.vector().get_local()

        f = Function(V_h)
        f.vector()[:] = x
        return f, x
