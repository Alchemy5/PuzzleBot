import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


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


def cloud_similarity(
    cloudA, cloudB
):  # already numpy arrays of n x d where n is num points and d is dimensions
    # import pdb
    # pdb.set_trace()

    # --- project to XY ---
    cloudA_xy = cloudA[:, :2]
    cloudB_xy = cloudB[:, :2]

    # --- normalize (center + scale to unit radius) ---
    for pts_name in ["A", "B"]:
        pts = cloudA_xy if pts_name == "A" else cloudB_xy
        pts -= pts.mean(axis=0, keepdims=True)
        max_r = np.linalg.norm(pts, axis=1).max()
        if max_r > 0:
            pts /= max_r
        if pts_name == "A":
            cloudA_xy = pts
        else:
            cloudB_xy = pts

    # --- build descriptors: [cov eigenvalues (2), radial histogram (32)] ---
    nbins = 32

    # A: covariance eigenvalues
    if cloudA_xy.shape[0] >= 2:
        covA = np.cov(cloudA_xy.T)
        eigsA, _ = np.linalg.eig(covA)
        eigsA = np.sort(np.real(eigsA))
    else:
        eigsA = np.array([0.0, 0.0], dtype=np.float64)

    # A: radial histogram
    rA = np.linalg.norm(cloudA_xy, axis=1)
    histA, _ = np.histogram(rA, bins=nbins, range=(0.0, 1.0), density=True)
    histA = histA.astype(np.float64)
    sA = histA.sum()
    if sA > 0:
        histA /= sA
    descA = np.concatenate([eigsA, histA])

    # B: covariance eigenvalues
    if cloudB_xy.shape[0] >= 2:
        covB = np.cov(cloudB_xy.T)
        eigsB, _ = np.linalg.eig(covB)
        eigsB = np.sort(np.real(eigsB))
    else:
        eigsB = np.array([0.0, 0.0], dtype=np.float64)

    # B: radial histogram
    rB = np.linalg.norm(cloudB_xy, axis=1)
    histB, _ = np.histogram(rB, bins=nbins, range=(0.0, 1.0), density=True)
    histB = histB.astype(np.float64)
    sB = histB.sum()
    if sB > 0:
        histB /= sB
    descB = np.concatenate([eigsB, histB])

    # --- cosine similarity in [0, 1] ---
    nA = np.linalg.norm(descA)
    nB = np.linalg.norm(descB)
    if nA == 0 or nB == 0:
        return 0.0

    sim = float(np.dot(descA, descB) / (nA * nB))
    # numerical guard: map [-1,1] → [0,1]
    sim = max(min(sim, 1.0), -1.0)
    sim = 0.5 * (sim + 1.0)

    return sim
