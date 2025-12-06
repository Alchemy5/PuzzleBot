# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes[0, 0].imshow(puzzle_color)
# axes[0, 0].set_title("Puzzle camera RGB")
# axes[0, 0].axis("off")

# im = axes[0, 1].imshow(puzzle_depth, cmap="magma")
# axes[0, 1].set_title("Puzzle camera depth")
# axes[0, 1].axis("off")
# fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

# axes[1, 0].imshow(tray_color)
# axes[1, 0].set_title("Tray camera RGB")
# axes[1, 0].axis("off")

# im = axes[1, 1].imshow(tray_depth, cmap="magma")
# axes[1, 1].set_title("Tray camera depth")
# axes[1, 1].axis("off")
# fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()

def crop_table(cloud, z_min=-0.025):
    xyz = np.asarray(cloud.xyzs())
    keep = xyz[2, :] > z_min
    new_cloud = type(cloud)(new_size=int(keep.sum()), fields=cloud.fields())
    new_cloud.mutable_xyzs()[:] = xyz[:, keep]
    if cloud.has_rgbs():
        new_cloud.mutable_rgbs()[:] = np.asarray(cloud.rgbs())[:, keep]
    return new_cloud
    # return cloud

# cropped_cloud = crop_table(full_puzzle_cloud, z_min=-0.025)

# def visualize_depth_and_gradient(depth: np.ndarray, step=20):
#     """
#     depth: HxW float array (meters). Assumes depth increases away from camera.
#     step: stride (pixels) for quiver sampling.
#     """
#     # Compute spatial gradients in image coords (v,u)
#     Gy, Gx = np.gradient(depth)  # Gy = d(depth)/dv, Gx = d(depth)/du

#     # Gradient magnitude
#     mag = np.hypot(Gx, Gy) + 1e-9
#     # Steepest descent directions (toward smaller depth)
#     dx = -Gx / mag
#     dy = -Gy / mag

#     H, W = depth.shape
#     u = np.arange(0, W, step)
#     v = np.arange(0, H, step)
#     uu, vv = np.meshgrid(u, v)

#     plt.figure(figsize=(10, 4))

#     # Depth image
#     plt.subplot(1, 2, 1)
#     im1 = plt.imshow(depth, cmap=cm.viridis)
#     plt.title("Depth (m)")
#     plt.colorbar(im1, fraction=0.046, pad=0.04)
#     plt.axis("off")

#     # Gradient magnitude + vectors
#     plt.subplot(1, 2, 2)
#     im2 = plt.imshow(mag, cmap=cm.magma)
#     plt.title("|∇ depth| with descent vectors")
#     plt.colorbar(im2, fraction=0.046, pad=0.04)
#     plt.quiver(uu, vv, dx[::step, ::step], dy[::step, ::step],
#                color="cyan", angles="xy", scale_units="xy", scale=0.8, width=0.003)
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt
from puzzle_config import (
    camera_translation,
    cross_translation,
    infinity_translation,
    lower_left_translation,
    lower_right_translation,
    my_piece_translation,
    puzzle_center,
    puzzle_center_x,
    puzzle_center_y,
    puzzle_center_z,
    puzzle_offset,
    rectangle_translation,
    trapezoid_translation,
    tray_camera_translation,
    tray_translations,
    upper_left_translation,
    upper_right_translation,
)

def plot_point_cloud(cloud, title="Puzzle cloud", stride=1, s=1.0):
    """
    cloud: pydrake.geometry.PointCloud
    stride: subsample factor to reduce points for plotting
    s: matplotlib marker size
    """
    xyz = np.asarray(cloud.xyzs())  # shape (3, N)
    xyz = xyz[:, ::stride]

    # Optional RGB
    colors = None
    if cloud.has_rgbs():
        rgb = np.asarray(cloud.rgbs())[:, ::stride].T / 255.0  # shape (N,3)
        colors = rgb

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[0], xyz[1], xyz[2], c=colors, s=s, linewidths=0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# plot_point_cloud(full_puzzle_cloud)

# import numpy as np
# import matplotlib.pyplot as plt

# def cloud_height_map(cloud, resolution=0.005):
#     xyz = np.asarray(cloud.xyzs())  # (3, N)
#     x, y, z = xyz
#     x_min, x_max = x.min(), x.max()
#     y_min, y_max = y.min(), y.max()

#     # xs = np.arange(x_min, x_max + resolution, resolution)
#     # ys = np.arange(y_min, y_max + resolution, resolution)
#     half_x = max(puzzle_center_x - x_min, x_max - puzzle_center_x)
#     half_y = max(puzzle_center_y - y_min, y_max - puzzle_center_y)
#     xs = np.arange(puzzle_center_x - half_x,
#                    puzzle_center_x + half_x + resolution, resolution)
#     ys = np.arange(puzzle_center_y - half_y,
#                    puzzle_center_y + half_y + resolution, resolution)
#     hmap = np.full((len(ys), len(xs)), np.nan)

#     i = np.floor((x - x_min) / resolution).astype(int)
#     j = np.floor((y - y_min) / resolution).astype(int)
#     for u, v, zz in zip(i, j, z):
#         if np.isnan(hmap[v, u]):
#             hmap[v, u] = zz
#         else:
#             hmap[v, u] = max(hmap[v, u], zz)
#     return hmap, xs, ys

def cloud_height_map(cloud, resolution=0.005):
    xyz = np.asarray(cloud.xyzs())
    x, y, z = xyz
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # half_x = resolution * np.ceil(max(puzzle_center_x - x_min, x_max - puzzle_center_x) / resolution)
    # half_y = resolution * np.ceil(max(puzzle_center_y - y_min, y_max - puzzle_center_y) / resolution)
    # xs = np.arange(puzzle_center_x - half_x, puzzle_center_x + half_x + resolution, resolution)
    # ys = np.arange(puzzle_center_y - half_y, puzzle_center_y + half_y + resolution, resolution)
    half_x = resolution * np.ceil(max(puzzle_center_x - x_min, x_max - puzzle_center_x) / resolution)
    half_y = resolution * np.ceil(max(puzzle_center_y - y_min, y_max - puzzle_center_y) / resolution)

    n_x = int(np.ceil(2 * half_x / resolution))
    if n_x % 2 == 0:
        n_x += 1  # force an odd count so the center sample exists
    n_y = int(np.ceil(2 * half_y / resolution))
    if n_y % 2 == 0:
        n_y += 1

    xs = puzzle_center_x + resolution * np.arange(-(n_x // 2), n_x // 2 + 1)
    ys = puzzle_center_y + resolution * np.arange(-(n_y // 2), n_y // 2 + 1)

    hmap = np.full((len(ys), len(xs)), np.nan)

    i = np.floor((x - xs[0]) / resolution).astype(int)
    j = np.floor((y - ys[0]) / resolution).astype(int)
    valid = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
    for u, v, zz in zip(i[valid], j[valid], z[valid]):
        hmap[v, u] = zz if np.isnan(hmap[v, u]) else max(hmap[v, u], zz)
    return hmap, xs, ys


def signed_distance_box(xx, yy, box_min, box_max):
    # box_min, box_max: (x_min, y_min), (x_max, y_max)
    dx = np.maximum(np.maximum(box_min[0] - xx, 0), xx - box_max[0])
    dy = np.maximum(np.maximum(box_min[1] - yy, 0), yy - box_max[1])
    outside = (dx > 0) | (dy > 0)
    dist_out = np.hypot(dx, dy)
    dist_in = -np.minimum(np.minimum(xx - box_min[0], box_max[0] - xx),
                          np.minimum(yy - box_min[1], box_max[1] - yy))
    return np.where(outside, dist_out, dist_in)

corners = np.array([
    upper_left_translation[:2],
    upper_right_translation[:2],
    lower_left_translation[:2],
    lower_right_translation[:2],
])
x_min, y_min = corners.min(axis=0)
x_max, y_max = corners.max(axis=0)
margin = 0.005

box_min = (x_min - margin, y_min - margin)
box_max = (x_max + margin, y_max + margin)

def visualize_height_and_gradient(cloud, resolution=0.005, step=1,
                                  box_min=box_min, box_max=box_max):
    hmap, xs, ys = cloud_height_map(cloud, resolution)
    if not np.isfinite(hmap).any():
        print("Height map is empty"); return
    mean_val = np.nanmean(hmap)
    hmap = np.where(np.isfinite(hmap), hmap, mean_val)

    kernel = np.ones((3, 3), dtype=hmap.dtype) / 9.0
    hmap = np.pad(hmap, 1, mode="edge")
    hmap = (
        hmap[:-2, :-2] + hmap[:-2, 1:-1] + hmap[:-2, 2:] +
        hmap[1:-1, :-2] + hmap[1:-1, 1:-1] + hmap[1:-1, 2:] +
        hmap[2:, :-2] + hmap[2:, 1:-1] + hmap[2:, 2:]
    ) / 9.0

    Gy, Gx = np.gradient(hmap, resolution, resolution)  # d/dy, d/dx
    xx, yy = np.meshgrid(xs, ys)
    phi = signed_distance_box(xx, yy, box_min, box_max)
    phiy, phix = np.gradient(phi, resolution, resolution)

    # Choose direction: outside -> inward to box; inside -> downhill height
    dir_x = np.where(phi > 0, -phix, -Gx)
    dir_y = np.where(phi > 0, -phiy, -Gy)
    # mag = np.hypot(dir_x, dir_y) + 1e-9
    # dir_x /= mag; dir_y /= mag

    arrow_scale = 0.01
    dx_draw, dy_draw = arrow_scale * dir_x, arrow_scale * dir_y

    plt.figure(figsize=(8, 6))
    # plt.imshow(hmap, origin="lower",
    #            extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="plasma")
    extent = [xs[0] - resolution / 2, xs[-1] + resolution / 2,
          ys[0] - resolution / 2, ys[-1] + resolution / 2]
    plt.imshow(hmap, origin="lower", extent=extent, cmap="plasma")
    plt.colorbar(label="Height (m)")
    plt.quiver(xx[::step, ::step], yy[::step, ::step],
               dx_draw[::step, ::step], dy_draw[::step, ::step],
               color="white", angles="xy", scale_units="xy", scale=1.0,
               width=0.002, headwidth=4, headlength=6, headaxislength=5,
               pivot="mid")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title("Height map with inward/outward push vectors")
    plt.tight_layout(); plt.show()

# Example:
# plot_point_cloud(cropped_cloud)
# visualize_height_and_gradient(full_puzzle_cloud, resolution=0.005, step=2)

# import numpy as np
# import matplotlib.pyplot as plt
# from puzzle_config import puzzle_center

# def visualize_push_to_center(cloud, resolution=0.005, step=2, xlim=None, ylim=None, padding=0.05):
#     # Rasterize heights just for a background heatmap
#     hmap, xs, ys = cloud_height_map(cloud, resolution)
#     if not np.isfinite(hmap).any():
#         print("Height map is empty"); return
#     hmap = np.where(np.isfinite(hmap), hmap, np.nanmean(hmap))

#     # Prepare grid
#     xx, yy = np.meshgrid(xs, ys)
#     cx, cy = puzzle_center[:2]

#     # Vectors pointing from each cell toward the center
#     dir_x = cx - xx
#     dir_y = cy - yy
#     mag = np.hypot(dir_x, dir_y) + 1e-9
#     dir_x /= mag; dir_y /= mag

#     arrow_scale = 0.01
#     dx_draw = arrow_scale * dir_x
#     dy_draw = arrow_scale * dir_y

#     x_min, x_max = xs[0], xs[-1]
#     y_min, y_max = ys[0], ys[-1]
#     if xlim is None: xlim = (x_min - padding, x_max + padding)
#     if ylim is None: ylim = (y_min - padding, y_max + padding)

#     plt.figure(figsize=(8, 6))
#     plt.imshow(hmap, origin="lower",
#                extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="viridis")
#     plt.xlim(xlim); plt.ylim(ylim)
#     plt.colorbar(label="Height (m)")
#     plt.quiver(xx[::step, ::step], yy[::step, ::step],
#                dx_draw[::step, ::step], dy_draw[::step, ::step],
#                color="white", angles="xy", scale_units="xy", scale=1.0,
#                width=0.002, headwidth=4, headlength=6, headaxislength=5,
#                pivot="mid")
#     plt.xlabel("X (m)"); plt.ylabel("Y (m)")
#     plt.title("Push vectors toward puzzle center")
#     plt.tight_layout()
#     plt.show()

# visualize_push_to_center(full_puzzle_cloud, resolution=0.009, step=1)
# visualize_depth_and_gradient(depth=puzzle_depth_crop)