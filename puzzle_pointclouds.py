import numpy as np
from pydrake.all import Context, Diagram, PointCloud
from typing import Dict, Mapping, Tuple

from puzzle_config import (
    puzzle_center as DEFAULT_PUZZLE_CENTER,
    tray_translations as DEFAULT_TRAY_TRANSLATIONS,
)

PUZZLE_HALF_EXTENTS = np.array([0.06, 0.06, 0.03]) 
TRAY_HALF_EXTENTS   = np.array([0.04, 0.04, 0.03])
def crop_aabb(pc: PointCloud, lower, upper) -> PointCloud:
    lower = np.asarray(lower).reshape(3, 1)
    upper = np.asarray(upper).reshape(3, 1)

    xyz = np.asarray(pc.xyzs())
    mask = np.all((xyz >= lower) & (xyz <= upper), axis=0)

    xyz_sel = xyz[:, mask]
    out = PointCloud(xyz_sel.shape[1], pc.fields())
    out.mutable_xyzs()[:] = xyz_sel

    if pc.has_rgbs():
        rgbs = np.asarray(pc.rgbs())
        rgbs_sel = rgbs[:, mask]
        out.mutable_rgbs()[:] = rgbs_sel

    return out


def get_puzzle_pointcloud(diagram: Diagram, context: Context) -> PointCloud:
    return diagram.GetOutputPort("puzzle.point_cloud").Eval(context)

def get_tray_pointcloud(diagram: Diagram, context: Context) -> PointCloud:
    return diagram.GetOutputPort("tray.point_cloud").Eval(context)

def get_puzzle_and_tray_pointclouds(
    diagram: Diagram,
    context: Context,
    puzzle_center=DEFAULT_PUZZLE_CENTER,
    tray_translations: Mapping[str, Tuple[float, float, float]] | None = None,
) -> Tuple[PointCloud, Dict[str, PointCloud]]:
    tray_translations = tray_translations or DEFAULT_TRAY_TRANSLATIONS

    puzzle_pc_full = get_puzzle_pointcloud(diagram, context)
    puzzle_center_vec = np.asarray(puzzle_center)
    puzzle_lower = puzzle_center_vec - PUZZLE_HALF_EXTENTS
    puzzle_upper = puzzle_center_vec + PUZZLE_HALF_EXTENTS
    puzzle_cloud = crop_aabb(puzzle_pc_full, puzzle_lower, puzzle_upper)

    print_pointcloud_bounds(puzzle_pc_full, "puzzle_full_cloud")
    print_pointcloud_bounds(puzzle_cloud, "puzzle_cropped_cloud")

    tray_pc_full = get_tray_pointcloud(diagram, context)
    print_pointcloud_bounds(tray_pc_full, "tray_full_cloud")

    tray_clouds: Dict[str, PointCloud] = {}

    def crop_piece(name: str, translation):
        center = np.asarray(translation)
        lower = center - TRAY_HALF_EXTENTS
        upper = center + TRAY_HALF_EXTENTS
        tray_pc = crop_aabb(tray_pc_full, lower, upper)
        tray_clouds[name] = tray_pc
        print_pointcloud_bounds(tray_pc, f"tray_{name}_cloud")

    for name, translation in tray_translations.items():
        crop_piece(name, translation)

    return puzzle_cloud, tray_clouds

def print_pointcloud_bounds(pc: PointCloud, name: str = "cloud") -> None:
    xyz = np.asarray(pc.xyzs())
    finite = np.isfinite(xyz).all(axis=0)
    xyz = xyz[:, finite]

    if xyz.shape[1] == 0:
        print(f"{name}: no finite points")
        return

    x_min, x_max = xyz[0].min(), xyz[0].max()
    y_min, y_max = xyz[1].min(), xyz[1].max()
    z_min, z_max = xyz[2].min(), xyz[2].max()

    print(f"{name} bounds:")
    print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  z: [{z_min:.3f}, {z_max:.3f}]")
