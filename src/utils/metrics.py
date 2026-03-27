"""
A collection of functions to compute reconstruction error metrics
"""
import numpy as np
from fenics import FunctionSpace, Point
from skimage.segmentation import chan_vese
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim


def relative_segmentation(x: np.ndarray, tau: float) -> np.ndarray:
    """Compute the relative threshold (fraction of max) segmentation."""
    return x >= tau * np.max(x)


def error_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return intersection / union


def error_auc_iou(
        x: np.ndarray, x_hat: np.ndarray, tau_range: np.ndarray = np.linspace(0.1, 1, 100)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x, np.array         : ground truth
    x_hat, np.array     : tikhonov solution
    tau_range, np.array : thresholds as fraction of max

    returns: (auc_iou, tau_max, ious)
    """
    ious = np.zeros(len(tau_range))

    for i, tau in enumerate(tau_range):
        mask = relative_segmentation(x, tau)
        mask_hat = relative_segmentation(x_hat, tau)
        ious[i] = error_iou(mask, mask_hat)

    return np.trapz(ious, tau_range), tau_range[np.argmax(ious)], ious


class SpaceIndexing:
    """Simple class which contains indexing and dimension info about the function space V_h."""
    def __init__(self, V_h: FunctionSpace):
        self.coords = V_h.tabulate_dof_coordinates()
        self.grid_indices = np.lexsort((self.coords[:, 0], self.coords[:, 1]))
        self.dof_indices = np.argsort(self.grid_indices)
        self.n = int(np.sqrt(V_h.dim()))


def matrix_to_vec(X, space: SpaceIndexing):
    return X.flatten()[space.dof_indices]


def vec_to_matrix(x, space: SpaceIndexing):
    return x[space.grid_indices].reshape((space.n, space.n))


def centroid(x):
    idx = np.arange(len(x))
    return np.sum(idx * x) / np.sum(x)


def error_centroid(x, x_hat):
    return abs(centroid(x) - centroid(x_hat))


def error_correlation(x, x_hat):
    corr = np.correlate(x, x_hat, mode='full')
    shift = np.argmax(corr) - (len(x)-1)
    return shift


def error_movers(x, x_hat):
    i = np.arange(len(x))
    dist = wasserstein_distance(i, i, np.abs(x), np.abs(x_hat))
    return dist


def rectangular_interpolation(mesh, f):
    """Interpolate f onto a square grid Z"""
    coords = mesh.coordinates()

    # Calculate number of x and y nodes
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    num_nodes = mesh.num_vertices()
    nx = ny = int(np.sqrt(num_nodes))
    nx, ny = int(nx*1.2), int(ny*1.2)

    # Construct mesh grid
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    # Interpolation
    Z = np.zeros_like(X)
    tree = mesh.bounding_box_tree()
    for j in range(ny):
        for i in range(nx):
            p = Point(X[j, i], Y[j, i])
            if tree.compute_first_entity_collision(p) < mesh.num_cells():
                Z[j, i] = f(p)  # evaluate f
            else:
                Z[j, i] = np.nan  # outside domain

    Z_norm = (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z))
    return Z_norm


def compute_cv_mask(X, mu=0.1, lambda1=1, lambda2=1):
    if np.isnan(X).any():
        X = np.nan_to_num(X, copy=True, nan=0.0)

    cv = chan_vese(X,
        mu=mu,        # contour length penalty (smoothness)
        lambda1=lambda1,      # weight for inside region
        lambda2=lambda2,      # weight for outside region
    )
    return cv
    

def error_ssim(X, X_hat):
    X = X.astype(float)
    X_hat = X_hat.astype(float)

    s1 = ssim(X, X_hat, data_range=1.0)
    s2 = ssim(X, 1 - X_hat, data_range=1.0)

    return max(s1, s2)
