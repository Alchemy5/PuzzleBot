import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree


def find_z_centers(points):
    z = points[:, 2]
    centers = np.array([z.min(), z.max()])

    for _ in range(10):
        labels = np.abs(z[:, None] - centers[None, :]).argmin(axis=1)

        centers = np.array([z[labels == k].mean() for k in (0, 1)])

    return np.sort(centers)


def find_closest_z_center(point, center1, center2):
    if np.abs(point[2] - center1) > np.abs(point[2] - center2):
        return center2
    else:
        return center1


def largest_region(points, radius=0.01):
    pts = np.asarray(points)

    # eps = radius, min_samples=1 ensures every point belongs to some cluster
    labels = DBSCAN(eps=radius, min_samples=1).fit_predict(pts)

    # Find label of the largest cluster
    uniq, counts = np.unique(labels, return_counts=True)
    best_label = uniq[np.argmax(counts)]

    return pts[labels == best_label]


def cloud_similarity(  # A is negative space and B is positive space
    cloudA, cloudB
):  # already numpy arrays of n x d where n is num points and d is dimensions
    def FindClosestPoints(query_points, reference_points):
        """
        Finds the nearest (Euclidean) neighbor in reference_points for each
        point in query_points. inputs are two n x 2 point cloud arrays
        """
        indices = np.empty(query_points.shape[0], dtype=int)

        kdtree = KDTree(reference_points)
        for i in range(query_points.shape[0]):
            distance, indices[i] = kdtree.query(query_points[i, :], k=1)

        return indices

    # project to xy helps since we only care about this plane
    cloudA_xy = cloudA[:, :2]
    cloudB_xy = cloudB[:, :2]

    max_iterations = 30
    tolerance = 1e-5

    R = np.eye(2)
    t = np.zeros(2)

    prev_error = np.inf

    for _ in range(max_iterations):
        B_xy_transformed = cloudB_xy @ R.T + t
        indices = FindClosestPoints(B_xy_transformed, cloudA_xy)
        A_match = cloudA_xy[indices]

        centroid_B = B_xy_transformed.mean(axis=0)
        centroid_A = A_match.mean(axis=0)

        # center at origin
        B_centered = B_xy_transformed - centroid_B
        A_centered = A_match - centroid_A

        H = B_centered.T @ A_centered

        U, S, Vt = np.linalg.svd(H)

        # incremental rotation that best aligns B to A
        R_delta = Vt.T @ U.T

        if np.linalg.det(R_delta) < 0:
            Vt[1, :] *= -1
            R_delta = Vt.T @ U.T

        # compute incremental translation best aligning B to A
        t_delta = centroid_A - centroid_B @ R_delta.T

        R = R_delta @ R
        t = R_delta @ t + t_delta

        B_xy_transformed = cloudB_xy @ R.T + t

        errors = np.linalg.norm(A_match - B_xy_transformed, axis=1)
        mean_error = errors.mean()

        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    alignedB = cloudB.copy()
    alignedB[:, :2] = cloudB_xy @ R.T + t

    similarity_score = 1.0 / (1e-6 + mean_error)
    return similarity_score, alignedB, R, t  # less similarity score is better
