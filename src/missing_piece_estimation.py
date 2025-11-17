import numpy as np


# Step 1: Given an almost finished puzzle, estimate shape of missing piece/cavity -> negative space detection
# Step 2: Given an array of point clouds, match the shape of the missing piece with one of the point clouds
# Step 3: Estimate pose of missing piece relative to initial pose of missing piece

# assuming missing puzzle point cloud is cropped such that flat table background is not present/only puzzle set point cloud data is present along with that of gap

# loop through point cloud data and identify two depths: one depth for the puzzle pieces and one depth for the areas without puzzle piece
# filter out all point clouds corresponding to d1 such that only left with d2

# use to compute outline of missing puzzle piece

# given convex shapes of other point clouds rotate them and compute similarity scores between the two based on icp principles

# how to work with point cloud?

# assume it is a 3 x N array of points is point cloud


# given an array with the values being close to either one of two central values how to find these two central values?
# k means with 2 clusters


def find_z_centers(points):
    z = points[:, 2]
    centers = np.array([z.min(), z.max()], dtype=float)

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
    pts = np.asarray(points, float)
    n = len(pts)
    if n == 0:
        return pts
    d = pts[:, None, :] - pts[None, :, :]
    A = np.sum(d * d, axis=-1) <= radius * radius
    vis = np.zeros(n, bool)
    best = []
    for i in range(n):
        if vis[i]:
            continue
        stack = [i]
        vis[i] = True
        comp = []
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in np.nonzero(A[j])[0]:
                if not vis[k]:
                    vis[k] = True
                    stack.append(k)
        if len(comp) > len(best):
            best = comp
    return pts[best]


"""
Old
def find_two_centers(x, max_iter=100, tol=1e-6, seed=0):
    x = np.asarray(x).ravel()  # ensure 1D
    rng = np.random.default_rng(seed)

    # 1. Initialize centers by picking two random points
    centers = rng.choice(x, size=2, replace=False)

    for _ in range(max_iter):
        old_centers = centers.copy()

        # 2. Assign each point to the nearest center
        # distances shape: (n_points, 2)
        distances = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        # 3. Recompute centers as the mean of assigned points
        for k in range(2):
            cluster_points = x[labels == k]
            if len(cluster_points) > 0:
                centers[k] = cluster_points.mean()

        # 4. Check for convergence
        if np.max(np.abs(centers - old_centers)) < tol:
            break

    # Sort centers so (center1 < center2) for consistency
    centers = np.sort(centers)
    return centers


rng = np.random.default_rng(seed=42)
N = 100

arr = rng.random((3, 100))
print(arr.T)
center1, center2 = 0.0, 1.0
negative_space_points = []

for point in arr.T:
    if np.abs(point[2] - center1) > np.abs(point[2] - center2):
    continue
	else:
    negative_space_points.append(point)

# do convex hull around this set of negative_space_points

# then try figuring out pose from this however what happens next heavily depends on what is given by Marik
"""
