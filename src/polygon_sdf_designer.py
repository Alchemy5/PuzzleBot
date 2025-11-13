#!/usr/bin/env python3
"""Interactive polygon designer that exports an extruded mesh as an SDF model.

Run with:
    python src/polygon_sdf_designer.py --thickness 0.01 --output assets/my_piece
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button

Point2D = Tuple[float, float]
Triangle = Tuple[int, int, int]


def polygon_area(points: Sequence[Point2D]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def ensure_ccw(points: List[Point2D]) -> List[Point2D]:
    if polygon_area(points) < 0:
        return list(reversed(points))
    return points


def point_in_triangle(pt: Point2D, tri: Tuple[Point2D, Point2D, Point2D]) -> bool:
    def sign(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, tri[0], tri[1])
    d2 = sign(pt, tri[1], tri[2])
    d3 = sign(pt, tri[2], tri[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def is_convex(prev_pt: Point2D, curr_pt: Point2D, next_pt: Point2D) -> bool:
    cross = ((curr_pt[0] - prev_pt[0]) * (next_pt[1] - curr_pt[1])
             - (curr_pt[1] - prev_pt[1]) * (next_pt[0] - curr_pt[0]))
    return cross > 0


def ear_clip_triangulation(points: Sequence[Point2D]) -> List[Triangle]:
    if len(points) < 3:
        raise ValueError("Need at least three points to triangulate.")

    polygon = list(points)
    vertex_indices = list(range(len(polygon)))
    triangles: List[Triangle] = []

    while len(vertex_indices) > 3:
        ear_found = False
        for i in range(len(vertex_indices)):
            prev_idx = vertex_indices[(i - 1) % len(vertex_indices)]
            curr_idx = vertex_indices[i]
            next_idx = vertex_indices[(i + 1) % len(vertex_indices)]

            prev_pt = polygon[prev_idx]
            curr_pt = polygon[curr_idx]
            next_pt = polygon[next_idx]

            if not is_convex(prev_pt, curr_pt, next_pt):
                continue

            tri_pts = (prev_pt, curr_pt, next_pt)
            contains_point = False
            for other_idx in vertex_indices:
                if other_idx in (prev_idx, curr_idx, next_idx):
                    continue
                if point_in_triangle(polygon[other_idx], tri_pts):
                    contains_point = True
                    break

            if contains_point:
                continue

            triangles.append((prev_idx, curr_idx, next_idx))
            del vertex_indices[i]
            ear_found = True
            break

        if not ear_found:
            raise ValueError("Failed to triangulate polygon. Ensure the points are simple and non-self-intersecting.")

    triangles.append(tuple(vertex_indices))
    return triangles


@dataclass
class PolygonDesigner:
    thickness: float
    output_prefix: Path
    model_name: str
    density: float
    overwrite: bool
    grid_step: float
    axis_extent: float
    points: List[Point2D] = field(default_factory=list)
    saved: bool = False

    def __post_init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect('equal', adjustable='box')
        self.title_text = self.ax.set_title(
            "Left-click to add vertices (snaps to grid), right-click to undo, Save when done"
        )
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_xlim(-self.axis_extent, self.axis_extent)
        self.ax.set_ylim(-self.axis_extent, self.axis_extent)
        self.scatter = self.ax.scatter([], [])
        (self.line,) = self.ax.plot([], [], '-o', color='tab:blue')

        self.status_text = self.fig.text(
            0.02,
            0.02,
            "0 vertices",
            transform=self.fig.transFigure,
            color="tab:gray",
        )

        save_ax = self.fig.add_axes([0.75, 0.02, 0.2, 0.06])
        self.save_button = Button(save_ax, 'Save', color='lightgreen', hovercolor='lightblue')
        self.save_button.on_clicked(self.on_save)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        if self.grid_step > 0:
            x = round(x / self.grid_step) * self.grid_step
            y = round(y / self.grid_step) * self.grid_step
        if event.button is MouseButton.LEFT:
            self.points.append((x, y))
        elif event.button is MouseButton.RIGHT:
            if self.points:
                self.points.pop()
        self.update_plot()

    def on_key_press(self, event) -> None:
        if event.key in {'enter', 'return'}:
            self.on_save(event)
        elif event.key in {'escape', 'q'}:
            plt.close(self.fig)

    def on_save(self, _event) -> None:
        if len(self.points) < 3:
            self.status_text.set_text("Need at least three vertices before saving.")
            self.status_text.set_color("tab:red")
            self.fig.canvas.draw_idle()
            return
        self.saved = True
        self.status_text.set_text("Saving shape...")
        self.status_text.set_color("tab:green")
        self.fig.canvas.draw_idle()
        print(f"Save clicked with {len(self.points)} vertices")
        plt.close(self.fig)

    def update_plot(self) -> None:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self.scatter.set_offsets(self.points)
        if self.points:
            loop_xs = xs + [self.points[0][0]]
            loop_ys = ys + [self.points[0][1]]
        else:
            loop_xs, loop_ys = [], []
        self.line.set_data(loop_xs, loop_ys)
        count = len(self.points)
        self.status_text.set_text(f"{count} vertex{'es' if count != 1 else ''}")
        self.status_text.set_color("tab:gray")
        self.fig.canvas.draw_idle()

    def get_polygon(self) -> List[Point2D]:
        unique_points: List[Point2D] = []
        for pt in self.points:
            if not unique_points or (abs(unique_points[-1][0] - pt[0]) > 1e-9 or abs(unique_points[-1][1] - pt[1]) > 1e-9):
                unique_points.append(pt)
        if len(unique_points) >= 3 and math.isclose(unique_points[0][0], unique_points[-1][0], abs_tol=1e-9) and math.isclose(unique_points[0][1], unique_points[-1][1], abs_tol=1e-9):
            unique_points.pop()
        return unique_points


def extrude_polygon(points: Sequence[Point2D], thickness: float) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    ccw_points = ensure_ccw(list(points))
    triangles = ear_clip_triangulation(ccw_points)
    half = thickness / 2.0
    vertices = [(x, y, -half) for x, y in ccw_points] + [(x, y, half) for x, y in ccw_points]

    faces: List[Tuple[int, int, int]] = []
    # Bottom faces (keep CCW as given)
    for tri in triangles:
        faces.append((tri[0] + 1, tri[1] + 1, tri[2] + 1))

    # Top faces (reverse to keep outward normal)
    for tri in triangles:
        faces.append((tri[2] + 1 + len(points), tri[1] + 1 + len(points), tri[0] + 1 + len(points)))

    # Side faces
    n = len(ccw_points)
    for i in range(n):
        j = (i + 1) % n
        b_i = i + 1
        b_j = j + 1
        t_i = i + 1 + n
        t_j = j + 1 + n
        faces.append((b_i, b_j, t_j))
        faces.append((b_i, t_j, t_i))

    return vertices, faces


def write_obj(path: Path, vertices: Sequence[Tuple[float, float, float]], faces: Sequence[Tuple[int, int, int]]) -> None:
    with path.open('w', encoding='ascii') as obj_file:
        for vx, vy, vz in vertices:
            obj_file.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for face in faces:
            obj_file.write(f"f {face[0]} {face[1]} {face[2]}\n")


def estimate_inertia(mass: float, points: Sequence[Point2D], thickness: float) -> Tuple[float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = thickness
    ixx = mass / 12.0 * (dy ** 2 + dz ** 2)
    iyy = mass / 12.0 * (dx ** 2 + dz ** 2)
    izz = mass / 12.0 * (dx ** 2 + dy ** 2)
    return ixx, iyy, izz


def write_sdf(
    path: Path,
    model_name: str,
    obj_filename: str,
    mass: float,
    inertia: Tuple[float, float, float],
    color: Tuple[float, float, float, float],
) -> None:
    ixx, iyy, izz = inertia
    ambient = f"{color[0] * 0.5:.3f} {color[1] * 0.5:.3f} {color[2] * 0.5:.3f} {color[3]:.3f}"
    diffuse = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} {color[3]:.3f}"
    specular = "0.2 0.2 0.2 1"
    sdf_contents = f"""<sdf version=\"1.7\">
  <model name=\"{model_name}\">
    <link name=\"{model_name}_link\">
      <inertial>
        <mass>{mass:.6f}</mass>
        <inertia>
          <ixx>{ixx:.6e}</ixx>
          <iyy>{iyy:.6e}</iyy>
          <izz>{izz:.6e}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name=\"visual\">
        <geometry>
          <mesh>
            <uri>{obj_filename}</uri>
          </mesh>
        </geometry>
                <material>
                    <ambient>{ambient}</ambient>
                    <diffuse>{diffuse}</diffuse>
                    <specular>{specular}</specular>
                    <emissive>0 0 0 1</emissive>
                </material>
      </visual>
      <collision name=\"collision\">
        <geometry>
          <mesh>
            <uri>{obj_filename}</uri>
          </mesh>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
    path.write_text(sdf_contents, encoding='ascii')


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive tool to draw a polygon and export an SDF.")
    parser.add_argument("--thickness", type=float, required=True, help="Extrusion thickness in meters.")
    parser.add_argument("--output", type=Path, required=True, help="Output path prefix (e.g. assets/my_piece).")
    parser.add_argument("--name", type=str, help="Optional model name. Defaults to the output file stem.")
    parser.add_argument("--density", type=float, default=500.0, help="Density in kg/m^3 used to estimate mass.")
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.01,
        help="Grid spacing in meters used when snapping clicks (set <=0 to disable).",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=0.2,
        help="Half-width of the drawing window in meters.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing files.")
    parser.add_argument(
        "--color",
        type=str,
        default="0.1,0.3,0.9,1.0",
        help="RGBA color for the piece, comma separated (default 0.1,0.3,0.9,1.0).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    output_prefix: Path = args.output
    output_dir = output_prefix.parent if output_prefix.parent != Path("") else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    designer = PolygonDesigner(
        thickness=args.thickness,
        output_prefix=output_prefix,
        model_name=args.name or output_prefix.stem,
        density=args.density,
        overwrite=args.overwrite,
        grid_step=args.grid_step,
        axis_extent=args.extent,
    )

    plt.show()

    if not designer.saved:
        print("No shape saved; exiting without creating files.")
        return 1

    polygon = designer.get_polygon()
    if len(polygon) < 3:
        print("Not enough distinct points to define a polygon; aborting.")
        return 1

    area = abs(polygon_area(polygon))
    mass = area * designer.thickness * designer.density

    try:
        vertices, faces = extrude_polygon(polygon, designer.thickness)
    except ValueError as exc:
        print(f"Error while building mesh: {exc}")
        return 1

    obj_path = output_prefix.with_suffix('.obj')
    sdf_path = output_prefix.with_suffix('.sdf')

    if not designer.overwrite:
        for path in (obj_path, sdf_path):
            if path.exists():
                print(f"{path} already exists. Use --overwrite to replace it.")
                return 1

    write_obj(obj_path, vertices, faces)

    inertia = estimate_inertia(mass, polygon, designer.thickness)
    color = tuple(float(c) for c in args.color.split(','))
    if len(color) != 4:
        print("Color must have four components (R,G,B,A).")
        return 1
    write_sdf(sdf_path, designer.model_name, obj_path.name, mass, inertia, color)

    print(f"Wrote {obj_path} and {sdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
