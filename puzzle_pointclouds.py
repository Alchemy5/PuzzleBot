import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from scipy.spatial import cKDTree
from pydrake.all import Context, Diagram, PointCloud
from pydrake.math import RollPitchYaw
from typing import Dict, Mapping, Tuple

from puzzle_config import (
    puzzle_center as DEFAULT_PUZZLE_CENTER,
    table_top_z,
    tray_translations as DEFAULT_TRAY_TRANSLATIONS,
)

PUZZLE_HALF_EXTENTS = np.array([0.08, 0.08, 0.04])
TRAY_SEARCH_RADIUS = 0.18
TRAY_CROP_MARGIN_XY = 0.025
TRAY_CROP_MARGIN_Z = 0.01
TRAY_MAX_ASSIGN_DISTANCE = 0.35
TRAY_FALLBACK_HALF_EXTENTS = np.array([0.15, 0.15, 0.05])
TRAY_COMPONENT_RADIUS = 0.025
TRAY_COMPONENT_ASSIGN_RADIUS = 0.05
TRAY_EXPECTED_MARGIN = np.array([0.005, 0.005, 0.005])
TRAY_EXPECTED_DIRECTIONAL_MARGIN: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "my_piece": (np.zeros(3), np.array([0.0, 0.03, 0.0])),
    "trapezoid": (np.zeros(3), np.array([0.03, 0.0, 0.0])),
}
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TRAY_SDF_FILENAMES = {
    "rectangle": "rectangle.sdf",
    "my_piece": "my_piece.sdf",
    "trapezoid": "trapezoid.sdf",
    "infinity": "infinity.sdf",
    "cross": "puzzle_cross.sdf",
}

# Height offset (in meters) used to strip table surface points while keeping most piece points.
DEFAULT_TABLE_CLEARANCE = 0.002


def _filter_points_above(
    pc: PointCloud,
    baseline_z: float,
    clearance: float,
    *,
    min_keep_fraction: float = 0.2,
) -> PointCloud:
    """Drops table points while keeping at least the top portion of the cloud."""
    xyz = np.asarray(pc.xyzs())
    if xyz.size == 0:
        return pc

    z_vals = xyz[2]
    total = xyz.shape[1]

    threshold = baseline_z + clearance
    mask = z_vals >= threshold
    keep = int(np.count_nonzero(mask))

    if keep == total:
        return pc

    min_keep = max(1, int(min_keep_fraction * total))
    if keep < min_keep:
        fallback_quantile = np.clip(1.0 - min_keep_fraction, 0.0, 1.0)
        threshold = np.quantile(z_vals, fallback_quantile)
        mask = z_vals >= threshold
        keep = int(np.count_nonzero(mask))
        if keep == 0:
            return pc

    out = PointCloud(keep, pc.fields())
    out.mutable_xyzs()[:] = xyz[:, mask]

    if pc.has_rgbs():
        rgbs = np.asarray(pc.rgbs())
        out.mutable_rgbs()[:] = rgbs[:, mask]

    return out


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


def _filter_by_assignment(pc: PointCloud, piece_index: int, translations_xy: np.ndarray) -> PointCloud:
    """Keeps only the points whose nearest tray translation matches the piece."""
    xyz = np.asarray(pc.xyzs())
    if xyz.size == 0:
        return pc

    xy = xyz[:2].T
    diff = xy[:, None, :] - translations_xy[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(dists.shape[0]), nearest]

    mask = nearest == piece_index
    if TRAY_MAX_ASSIGN_DISTANCE is not None:
        mask = mask & (min_dists <= TRAY_MAX_ASSIGN_DISTANCE)
        if not mask.any():
            mask = nearest == piece_index

    keep = int(np.count_nonzero(mask))
    if keep == 0:
        return PointCloud(0, pc.fields())
    if keep == mask.size:
        return pc

    out = PointCloud(keep, pc.fields())
    out.mutable_xyzs()[:] = xyz[:, mask]
    if pc.has_rgbs():
        out.mutable_rgbs()[:] = np.asarray(pc.rgbs())[:, mask]
    return out


def _parse_pose(text: str | None) -> Tuple[np.ndarray, np.ndarray]:
    if not text:
        return np.zeros(3), np.eye(3)
    values = [float(v) for v in text.strip().split()]
    if len(values) < 3:
        translation = np.zeros(3)
    else:
        translation = np.array(values[:3], dtype=float)
    if len(values) >= 6:
        rpy = values[3:6]
    else:
        rpy = [0.0, 0.0, 0.0]
    rotation = RollPitchYaw(*rpy).ToRotationMatrix().matrix()
    return translation, rotation


def _compose_transforms(
    parent_translation: np.ndarray,
    parent_rotation: np.ndarray,
    child_translation: np.ndarray,
    child_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    composed_rotation = parent_rotation @ child_rotation
    composed_translation = parent_translation + parent_rotation @ child_translation
    return composed_translation, composed_rotation


def _mesh_vertices_from_uri(sdf_path: Path, mesh_elem: ET.Element) -> np.ndarray:
    uri_text = mesh_elem.findtext("uri")
    if not uri_text:
        return np.empty((0, 3))
    mesh_path = (sdf_path.parent / uri_text.strip()).resolve()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        return vertices
    scale_text = mesh_elem.findtext("scale")
    if scale_text:
        scale = np.array([float(v) for v in scale_text.strip().split()], dtype=float)
        vertices = vertices * scale
    return vertices


def _geometry_vertices(sdf_path: Path, geometry: ET.Element) -> np.ndarray:
    mesh_elem = geometry.find("mesh")
    if mesh_elem is not None:
        return _mesh_vertices_from_uri(sdf_path, mesh_elem)

    box_elem = geometry.find("box")
    if box_elem is not None:
        size_text = box_elem.findtext("size")
        if size_text:
            size = np.array([float(v) for v in size_text.strip().split()], dtype=float)
            offsets = np.array([
                [sx, sy, sz]
                for sx in (-0.5 * size[0], 0.5 * size[0])
                for sy in (-0.5 * size[1], 0.5 * size[1])
                for sz in (-0.5 * size[2], 0.5 * size[2])
            ])
            return offsets
    return np.empty((0, 3))


@lru_cache(None)
def _get_tray_model_bounds(name: str) -> Tuple[np.ndarray, np.ndarray]:
    sdf_filename = TRAY_SDF_FILENAMES.get(name)
    if sdf_filename is None:
        raise KeyError(f"No SDF filename registered for tray piece '{name}'")

    sdf_path = ASSETS_DIR / sdf_filename
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    lower = np.array([np.inf, np.inf, np.inf], dtype=float)
    upper = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

    for collision in root.findall(".//collision"):
        collision_translation, collision_rotation = _parse_pose(collision.findtext("pose"))
        geometry = collision.find("geometry")
        if geometry is None:
            continue

        geometry_translation, geometry_rotation = _parse_pose(geometry.findtext("pose"))
        vertices = _geometry_vertices(sdf_path, geometry)
        if vertices.size == 0:
            continue

        total_translation, total_rotation = _compose_transforms(
            collision_translation,
            collision_rotation,
            geometry_translation,
            geometry_rotation,
        )

        transformed = (total_rotation @ vertices.T).T + total_translation
        lower = np.minimum(lower, transformed.min(axis=0))
        upper = np.maximum(upper, transformed.max(axis=0))

    if not np.all(np.isfinite(lower)):
        raise ValueError(f"Unable to compute bounds for tray piece '{name}' from {sdf_path}")

    return lower, upper


def _aabb_intersects(lower_a: np.ndarray, upper_a: np.ndarray, lower_b: np.ndarray, upper_b: np.ndarray) -> bool:
    return bool(np.all(upper_a >= lower_b) and np.all(upper_b >= lower_a))


def _connected_components(xyz: np.ndarray, radius: float) -> Tuple[np.ndarray, int]:
    """Labels connected components using a fixed-radius neighbor search."""
    if xyz.size == 0:
        return np.empty(0, dtype=int), 0

    count = xyz.shape[1]
    tree = cKDTree(xyz.T)
    labels = -np.ones(count, dtype=int)
    stack: list[int] = []
    component = 0

    for idx in range(count):
        if labels[idx] != -1:
            continue
        labels[idx] = component
        stack.append(idx)
        while stack:
            current = stack.pop()
            neighbors = tree.query_ball_point(xyz[:, current], radius)
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = component
                    stack.append(neighbor)
        component += 1

    return labels, component


def _filter_by_components(
    pc: PointCloud,
    component_labels,
    filtered_xyz: np.ndarray,
    filtered_labels: np.ndarray,
    filtered_tree: cKDTree | None,
) -> PointCloud:
    """Keeps only the points whose nearest filtered neighbor shares a component."""
    if filtered_tree is None or pc.size() == 0:
        return pc

    labels_array = np.array(list(component_labels), dtype=int)
    if labels_array.size == 0:
        return PointCloud(0, pc.fields())

    xyz = np.asarray(pc.xyzs())
    query_points = xyz.T
    distances, indices = filtered_tree.query(
        query_points,
        distance_upper_bound=TRAY_COMPONENT_ASSIGN_RADIUS,
    )

    valid = np.isfinite(distances)
    valid &= indices < filtered_labels.size
    if not valid.any():
        return PointCloud(0, pc.fields())

    allowed = np.zeros(valid.shape[0], dtype=bool)
    allowed_valid = np.isin(filtered_labels[indices[valid]], labels_array)
    allowed[valid] = allowed_valid

    keep = int(np.count_nonzero(allowed))
    if keep == 0:
        return PointCloud(0, pc.fields())
    if keep == allowed.size:
        return pc

    out = PointCloud(keep, pc.fields())
    out.mutable_xyzs()[:] = xyz[:, allowed]
    if pc.has_rgbs():
        out.mutable_rgbs()[:] = np.asarray(pc.rgbs())[:, allowed]
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
    min_height_above_table: float | None = DEFAULT_TABLE_CLEARANCE,
) -> Tuple[PointCloud, Dict[str, PointCloud]]:
    tray_translations = tray_translations or DEFAULT_TRAY_TRANSLATIONS

    clearance = None
    if min_height_above_table is not None:
        clearance = float(min_height_above_table)

    puzzle_pc_full = get_puzzle_pointcloud(diagram, context)
    puzzle_center_vec = np.asarray(puzzle_center)
    puzzle_lower = puzzle_center_vec - PUZZLE_HALF_EXTENTS
    puzzle_upper = puzzle_center_vec + PUZZLE_HALF_EXTENTS
    puzzle_cloud = crop_aabb(puzzle_pc_full, puzzle_lower, puzzle_upper)

    print_pointcloud_bounds(puzzle_pc_full, "puzzle_full_cloud")
    print_pointcloud_bounds(puzzle_cloud, "puzzle_cropped_cloud")

    if clearance is not None:
        puzzle_cloud = _filter_points_above(puzzle_cloud, table_top_z, clearance)
        print_pointcloud_bounds(puzzle_cloud, "puzzle_filtered_cloud")

    tray_pc_full = get_tray_pointcloud(diagram, context)
    print_pointcloud_bounds(tray_pc_full, "tray_full_cloud")

    tray_clouds: Dict[str, PointCloud] = {}

    tray_pc_for_cropping = tray_pc_full
    if clearance is not None:
        tray_pc_filtered_full = _filter_points_above(tray_pc_full, table_top_z, clearance)
        print_pointcloud_bounds(tray_pc_filtered_full, "tray_full_filtered_cloud")
        tray_pc_for_cropping = tray_pc_filtered_full

    names = list(tray_translations.keys())
    translations_array = np.array(
        [np.asarray(tray_translations[name], dtype=float) for name in names]
    )
    translations_xy = translations_array[:, :2]
    name_to_index = {name: idx for idx, name in enumerate(names)}

    expected_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, translation in zip(names, translations_array):
        local_lower, local_upper = _get_tray_model_bounds(name)
        expected_lower = translation + local_lower - TRAY_EXPECTED_MARGIN
        expected_upper = translation + local_upper + TRAY_EXPECTED_MARGIN

        directional = TRAY_EXPECTED_DIRECTIONAL_MARGIN.get(name)
        if directional is not None:
            lower_pad, upper_pad = directional
            expected_lower = expected_lower - lower_pad
            expected_upper = expected_upper + upper_pad

        expected_lower[2] = min(expected_lower[2], table_top_z - TRAY_CROP_MARGIN_Z)
        expected_upper[2] = max(expected_upper[2], table_top_z + local_upper[2] + TRAY_CROP_MARGIN_Z)
        expected_bounds[name] = (expected_lower, expected_upper)

    tray_xyz_filtered = np.asarray(tray_pc_for_cropping.xyzs())
    filtered_labels, num_components = _connected_components(
        tray_xyz_filtered, TRAY_COMPONENT_RADIUS
    )
    filtered_tree = None
    if tray_xyz_filtered.size:
        filtered_tree = cKDTree(tray_xyz_filtered.T)

    components: Dict[int, Dict[str, np.ndarray]] = {}
    for component_label in range(num_components):
        mask = filtered_labels == component_label
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        pts = tray_xyz_filtered[:, indices]
        lower_base = np.array([
            pts[0].min(),
            pts[1].min(),
            min(pts[2].min(), table_top_z),
        ])
        upper_base = np.array([
            pts[0].max(),
            pts[1].max(),
            pts[2].max(),
        ])
        components[component_label] = {
            "indices": indices,
            "lower_base": lower_base,
            "upper_base": upper_base,
        }

    components_by_piece: Dict[str, list[int]] = {name: [] for name in names}
    for label, data in components.items():
        comp_lower = data["lower_base"]
        comp_upper = data["upper_base"]
        for piece_name in names:
            expected_lower, expected_upper = expected_bounds[piece_name]
            if _aabb_intersects(comp_lower, comp_upper, expected_lower, expected_upper):
                components_by_piece[piece_name].append(label)

    def crop_piece(name: str) -> None:
        piece_index = name_to_index[name]
        translation = translations_array[piece_index]
        component_labels = components_by_piece.get(name, [])

        expected_lower, expected_upper = expected_bounds[name]
        lower = expected_lower.copy()
        upper = expected_upper.copy()

        subset_xyz = None
        if component_labels:
            index_groups = [components[label]["indices"] for label in component_labels if label in components]
            if index_groups:
                indices = np.concatenate(index_groups)
                if indices.size:
                    subset_xyz = tray_xyz_filtered[:, indices]

        if subset_xyz is not None and subset_xyz.size:
            observed_lower = np.array([
                subset_xyz[0].min() - TRAY_CROP_MARGIN_XY,
                subset_xyz[1].min() - TRAY_CROP_MARGIN_XY,
                min(subset_xyz[2].min(), table_top_z) - TRAY_CROP_MARGIN_Z,
            ])
            observed_upper = np.array([
                subset_xyz[0].max() + TRAY_CROP_MARGIN_XY,
                subset_xyz[1].max() + TRAY_CROP_MARGIN_XY,
                subset_xyz[2].max() + TRAY_CROP_MARGIN_Z,
            ])
            lower = np.minimum(lower, observed_lower)
            upper = np.maximum(upper, observed_upper)
        else:
            xyz_source = tray_xyz_filtered if tray_xyz_filtered.size else np.asarray(tray_pc_full.xyzs())
            if xyz_source.size:
                delta = xyz_source[:2] - translation[:2].reshape(2, 1)
                distances = np.linalg.norm(delta, axis=0)
                mask = distances <= TRAY_SEARCH_RADIUS
                if not mask.any():
                    nearest_count = max(1, int(0.05 * distances.size))
                    nearest_indices = np.argsort(distances)[:nearest_count]
                    mask = np.zeros(distances.shape, dtype=bool)
                    mask[nearest_indices] = True

                subset_xyz = xyz_source[:, mask]
                observed_lower = np.array([
                    subset_xyz[0].min() - TRAY_CROP_MARGIN_XY,
                    subset_xyz[1].min() - TRAY_CROP_MARGIN_XY,
                    min(subset_xyz[2].min(), table_top_z) - TRAY_CROP_MARGIN_Z,
                ])
                observed_upper = np.array([
                    subset_xyz[0].max() + TRAY_CROP_MARGIN_XY,
                    subset_xyz[1].max() + TRAY_CROP_MARGIN_XY,
                    subset_xyz[2].max() + TRAY_CROP_MARGIN_Z,
                ])
                lower = np.minimum(lower, observed_lower)
                upper = np.maximum(upper, observed_upper)
            else:
                lower = np.minimum(lower, translation - TRAY_FALLBACK_HALF_EXTENTS)
                upper = np.maximum(upper, translation + TRAY_FALLBACK_HALF_EXTENTS)

        tray_pc = crop_aabb(tray_pc_full, lower, upper)

        if clearance is not None:
            tray_pc = _filter_points_above(tray_pc, table_top_z, clearance)

        if component_labels:
            tray_pc = _filter_by_components(
                tray_pc,
                component_labels,
                tray_xyz_filtered,
                filtered_labels,
                filtered_tree,
            )
        else:
            tray_pc = _filter_by_assignment(tray_pc, piece_index, translations_xy)

        print_pointcloud_bounds(tray_pc, f"tray_{name}_cloud")
        tray_clouds[name] = tray_pc

    for name in names:
        crop_piece(name)

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
