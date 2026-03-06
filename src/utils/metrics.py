"""
A collection of functions to compute reconstruction error metrics
"""
import numpy as np
from fenics import FunctionSpace
from skimage.segmentation import chan_vese
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim


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


def compute_mask(X, mu=0.1, lambda1=1, lambda2=1):
    cv = chan_vese(X,
        mu=mu,        # contour length penalty (smoothness)
        lambda1=lambda1,      # weight for inside region
        lambda2=lambda2,      # weight for outside region
    )
    segmentation = cv
    return segmentation


def error_iou(x, x_hat, space: SpaceIndexing):
    X = vec_to_matrix(x, space=space)
    X_hat = vec_to_matrix(x_hat, space=space)
    mask = compute_mask(X)
    mask_hat = compute_mask(X_hat)
    iou = np.sum(mask & mask_hat) / np.sum(mask | mask_hat)
    return iou


def error_ssim(x, x_hat, space: SpaceIndexing):
    X = vec_to_matrix(x, space=space)
    X_hat = vec_to_matrix(x_hat, space=space)

    # Normalize
    X = (X - X.min()) / (X.max() - X.min())
    X_hat = (X_hat - X_hat.min()) / (X_hat.max() - X_hat.min())
    return ssim(X, X_hat, data_range=1.0)
