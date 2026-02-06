"""
Implementation of the randomized singular value decomposition.
"""
from numpy.random import randn
from scipy.linalg import svd, qr


def rsvd(A, k, p=5):
    """
    Given an m by n matrix A, a target rank k, and an oversampling parameter p,
    compute the approximate rank-k factorization `U S V^T` of A using the
    proto algorithm.
    """
    m, n = A.shape

    # Random matrix from a standard Gaussian distribution
    Omega = randn(n, min(k + p, m))

    # Obtain the n x (k + p) matrix Y = (A)(Omega)
    Y = A @ Omega

    # Get the orthonormal basis Q of Y
    Q, _ = qr(Y, mode='economic', check_finite=False)

    # Get the projection B of A onto span(Q)
    B = Q.T @ A

    # Compute the SVD of B
    U_B, S, Vt = svd(B, full_matrices=False, lapack_driver="gesdd", check_finite=False)

    # Project U_B back to the original space
    U = Q @ U_B

    # Return the k first vectors of the SVD
    return U[:, :k], S[:k], Vt[:k, :]
