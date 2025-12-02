from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    InverseKinematics,
    RotationMatrix,
    Solve,
    RigidTransform,
    Rgba,
)

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds,
)
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import trimesh
from controller import Controller, DepthController, WaypointPDController, PushToCenterController, KeepWsgOpen, PushAlongPhiController
from matplotlib import cm
from pydrake.systems.primitives import ConstantVectorSource

from puzzle_pointclouds import (
    get_puzzle_and_tray_pointclouds,
    get_puzzle_pointcloud,
    get_tray_pointcloud,
)
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


def _format_vec(vec: tuple[float, float, float]) -> str:
    return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"

def get_hardcoded_initial_gripper_pose(plant, plant_context, p_WG_des):
    # desired gripper pose, hover directly above the cross piece
    # hover_height = 0.20 
    # p_WG_des = np.array(cross_translation) + np.array([0.0, 0.0, hover_height])
    ik = InverseKinematics(plant, plant_context)
    q = ik.q()

    R_WG_des = RotationMatrix.MakeXRotation(-np.pi / 2)
    X_WG_des = RigidTransform(R_WG_des, p_WG_des)

    W = plant.world_frame()
    wsg_model = plant.GetModelInstanceByName("wsg")
    G = plant.GetBodyByName("body", wsg_model).body_frame()

    ik.AddPositionConstraint(
        G, # frameB
        [0.0, 0.0, 0.0], # p_BQ
        W, # frameA
        X_WG_des.translation() - 1e-3, # p_AQ_lower
        X_WG_des.translation() - 1e-3, # p_AQ_upper
    )

    ik.AddOrientationConstraint(
        W, # frameAbar
        R_WG_des, # R_AbarA
        G, # frameBbar
        RotationMatrix(), # R_BbarB
        1e-3, # theta_bound
    )

    # small quadratic cost to keep solution well-behaved
    prog = ik.prog()
    prog.AddQuadraticErrorCost(np.eye(len(q)), np.zeros(len(q)), q)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("ik failed to find a hover configuration")
    
    return result.GetSolution(q)


# def ik_to_hover_pose(plant, base_context, p_WG_des):
#     """solve ik for a top-down gripper at p_WG_des (world)."""
#     ik = InverseKinematics(plant, plant_context)
#     q = ik.q()

#     R_WG_des = RotationMatrix.MakeXRotation(-np.pi / 2)
#     X_WG_des = RigidTransform(R_WG_des, p_WG_des)

#     W = plant.world_frame()
#     wsg_model = plant.GetModelInstanceByName("wsg")
#     G = plant.GetBodyByName("body", wsg_model).body_frame()

#     ik.AddPositionConstraint(
#         G, # frameB
#         [0.0, 0.0, 0.0], # p_BQ
#         W, # frameA
#         X_WG_des.translation() - 1e-3, # p_AQ_lower
#         X_WG_des.translation() + 1e-3, # p_AQ_upper
#     )

#     ik.AddOrientationConstraint(
#         W, # frameAbar
#         X_WG_des.rotation(), # R_AbarA
#         G, # frameBbar
#         RotationMatrix(), # R_BbarB
#         0.0001 # theta_bound
#     )

#     # small quadratic cost to keep solution well-behaved
#     prog = ik.prog()
#     prog.AddQuadraticErrorCost(np.eye(len(q)), np.zeros(len(q)), q)

#     result = Solve(prog)
#     if not result.is_success():
#         raise RuntimeError("ik failed to find a hover configuration")
    
#     return result.GetSolution(q)

def ik_to_hover_pose(plant, base_context, p_WG_des):
    ik = InverseKinematics(plant, base_context.Clone())
    q = ik.q()

    X_WG_des = RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), p_WG_des)
    W = plant.world_frame()
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()

    ik.AddPositionConstraint(G, [0, 0, 0], W,
                             X_WG_des.translation() - 1e-3,
                             X_WG_des.translation() + 1e-3)
    ik.AddOrientationConstraint(W, X_WG_des.rotation(), G, RotationMatrix(), 1e-2)  # looser tol helps

    prog = ik.prog()
    prog.SetInitialGuess(q, plant.GetPositions(base_context))
    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("ik failed to find a hover configuration")
    return result.GetSolution(q)


# Start meshcat for visualization
meshcat = StartMeshcat()
print("Click the link above to open Meshcat in your browser!")


repo_root = Path(__file__).resolve().parent
assets_dir = repo_root / "assets"

# assets for tray pieces
my_piece_sdf_uri = (assets_dir / "my_piece.sdf").resolve().as_uri()
rectangle_sdf_uri = (assets_dir / "rectangle.sdf").resolve().as_uri()
trapezoid_sdf_uri = (assets_dir / "trapezoid.sdf").resolve().as_uri()
infinity_sdf_uri = (assets_dir / "infinity.sdf").resolve().as_uri()

# assets for welded puzzle frame
corner_sdf_uri = (assets_dir / "puzzle_corner.sdf").resolve().as_uri()
cross_sdf_uri = (assets_dir / "puzzle_cross.sdf").resolve().as_uri()


scenario_string = f"""directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
    default_joint_positions:
      iiwa_joint_1: [-1.57]
      iiwa_joint_2: [0.1]
      iiwa_joint_3: [0]
      iiwa_joint_4: [-1.2]
      iiwa_joint_5: [0]
      iiwa_joint_6: [1.6]
      iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0

- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}

- add_model:
    name: table
    file: "{(repo_root / 'table.sdf').resolve().as_uri()}"
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}


- add_model:
    name: custom_rectangle
    file: "{rectangle_sdf_uri}"
- add_weld:
    parent: world
    child: custom_rectangle::my_piece_link
    X_PC:
        translation: {_format_vec(rectangle_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}
- add_model:
    name: custom_my_piece
    file: "{my_piece_sdf_uri}"
- add_weld:
    parent: world
    child: custom_my_piece::my_piece_link
    X_PC:
        translation: {_format_vec(my_piece_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_model:
    name: trapezoid
    file: "{trapezoid_sdf_uri}"
- add_weld:
    parent: world
    child: trapezoid::trapezoid_link
    X_PC:
        translation: {_format_vec(trapezoid_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_model:
    name: infinity
    file: "{infinity_sdf_uri}"
- add_weld:
    parent: world
    child: infinity::infinity_link
    X_PC:
        translation: {_format_vec(infinity_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_model:
    name: puzzle_upper_right
    file: "{corner_sdf_uri}"
- add_weld:
    parent: world
    child: puzzle_upper_right::corner_link
    X_PC:
        translation: {_format_vec(upper_right_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}
- add_model:
    name: puzzle_upper_left
    file: "{corner_sdf_uri}"
- add_weld:
    parent: world
    child: puzzle_upper_left::corner_link
    X_PC:
        translation: {_format_vec(upper_left_translation)}
        rotation: !Rpy {{ deg: [0, 0, 90] }}
- add_model:
    name: puzzle_lower_left
    file: "{corner_sdf_uri}"
- add_weld:
    parent: world
    child: puzzle_lower_left::corner_link
    X_PC:
        translation: {_format_vec(lower_left_translation)}
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: puzzle_lower_right
    file: "{corner_sdf_uri}"
- add_weld:
    parent: world
    child: puzzle_lower_right::corner_link
    X_PC:
        translation: {_format_vec(lower_right_translation)}
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: puzzle_cross
    file: "{cross_sdf_uri}"
    default_free_body_pose:
        cross_link:
            translation: [0.1, -0.6, -0.015]
            rotation: !Rpy {{ deg: [0, 0, -90] }}

- add_model:
    name: puzzle_camera
    file: "package://manipulation/camera_box.sdf"
- add_weld:
    parent: world
    child: puzzle_camera::base
    X_PC:
        translation: {_format_vec(camera_translation)}
        rotation: !Rpy {{ deg: [180, 0, 0] }}

- add_model:
    name: tray_camera
    file: "package://manipulation/camera_box.sdf"
- add_weld:
    parent: world
    child: tray_camera::base
    X_PC:
        translation: {_format_vec(tray_camera_translation)}
        rotation: !Rpy {{ deg: [-150, 0, -10] }}

cameras:
  puzzle_camera:
    name: camera_puzzle
    depth: true
    X_PB:
        base_frame: puzzle_camera::base

  tray_camera:
    name: camera_tray
    depth: true
    X_PB:
        base_frame: tray_camera::base

"""
scenario = LoadScenario(data=scenario_string)
station = MakeHardwareStation(scenario, meshcat=meshcat)

builder = DiagramBuilder()
station_sys = builder.AddSystem(station)

"""
- add_weld:
    parent: world
    child: puzzle_cross::cross_link
    X_PC:
        translation: {_format_vec(cross_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}
"""

pcd_systems = AddPointClouds(builder=builder, station=station_sys, scenario=scenario)
print("Point cloud streams available:", list(pcd_systems.keys()))

# Expect keys "camera_puzzle" and "camera_tray" matching the scenario names.
expected_cloud_keys = {"camera_puzzle", "camera_tray"}
missing_clouds = expected_cloud_keys.difference(pcd_systems.keys())
if missing_clouds:
    raise KeyError(
        f"Missing expected point cloud streams: {sorted(missing_clouds)}."
        f" Available streams: {sorted(pcd_systems.keys())}"
    )

puzzle_pcd_sys = pcd_systems["camera_puzzle"]
tray_pcd_sys = pcd_systems["camera_tray"]

puzzle_pcd_port = puzzle_pcd_sys.point_cloud_output_port()
tray_pcd_port = tray_pcd_sys.point_cloud_output_port()

builder.ExportOutput(puzzle_pcd_port, "puzzle.point_cloud")
builder.ExportOutput(tray_pcd_port, "tray.point_cloud")


plant = station.GetSubsystemByName("plant")
plant_context = plant.CreateDefaultContext()
# iiwa_model = plant.GetModelInstanceByName("iiwa")
# plant_context = plant.CreateDefaultContext()

# waypoints_W = [
#     puzzle_center,
# #     cross_translation + np.array([0.0, 0.0, 0.20]),
# #     cross_translation + np.array([0.0, 0.0, 0.20]),
# ]

# q_waypoints = []
# for p_W in waypoints_W:
#     q_wp = ik_to_hover_pose(plant, plant_context, p_W)
#     q_waypoints.append(q_wp)

# q_waypoints = np.vstack(q_waypoints)
# controller = builder.AddSystem(WaypointPDController(plant, iiwa_model, q_waypoints))

# builder.Connect(
#     station.GetOutputPort("iiwa_state"),
#     controller.state_port,
# )
# builder.Connect(
#     controller.get_output_port(0),
#     station.GetInputPort("iiwa_actuation"),
# )

# plant_context = plant.CreateDefaultContext()
# ik = InverseKinematics(plant, plant_context)
# q = ik.q()
# q_initial = get_hardcoded_initial_gripper_pose(plant, plant_context, cross_translation)
# plant.SetDefaultPositions(q_initial)

# controller = builder.AddSystem(Controller(q_desired=q_initial))
# controller = builder.AddSystem(DepthController(plant))

# builder.Connect(
#     station.GetOutputPort("camera_puzzle.depth_image"),
#     controller.depth_port,
# )
# builder.Connect(
#     station.GetOutputPort("iiwa_generalized_contact_forces"),
#     controller.contact_port,
# )
# builder.Connect(
#     station.GetOutputPort("iiwa_state"),  # or similar state port
#     controller.get_input_port(0),
# )
# builder.Connect(
#     controller.get_output_port(0),
#     station.GetInputPort("iiwa_actuation"),
# )

# controller = builder.AddSystem(PushToCenterController(plant))
# builder.Connect(
#     station.GetOutputPort("iiwa_state"),
#     controller.get_input_port(0)
# )
# builder.Connect(
#     controller.get_output_port(0),
#     station.GetInputPort("iiwa_actuation"),
# )

wsg_ctrl = builder.AddSystem(KeepWsgOpen(target_width=0.01))
builder.Connect(station.GetOutputPort("wsg_state"), wsg_ctrl.state_port)
builder.Connect(wsg_ctrl.get_output_port(0), station.GetInputPort("wsg_actuation"))

phi_ctrl = builder.AddSystem(PushAlongPhiController(plant))
builder.Connect(station.GetOutputPort("iiwa_state"), phi_ctrl.state_port)
builder.Connect(phi_ctrl.get_output_port(0), station.GetInputPort("iiwa_actuation"))

grasp_pos = cross_translation + np.array([0, 0.04, 0.03])
q_grasp = ik_to_hover_pose(plant, plant_context, grasp_pos)

plant.SetDefaultPositions(q_grasp)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.ForcedPublish(diagram_context)

full_puzzle_cloud = get_puzzle_pointcloud(diagram, diagram_context)
full_tray_cloud = get_tray_pointcloud(diagram, diagram_context)

phi_ctrl.initialize_field(full_puzzle_cloud)

puzzle_cloud, tray_clouds = get_puzzle_and_tray_pointclouds(
    diagram,
    diagram_context,
    puzzle_center=puzzle_center,
    tray_translations=tray_translations,
)

print("Puzzle camera cloud has", full_puzzle_cloud.size(), "points")
print("Tray camera cloud has", full_tray_cloud.size(), "points")
print("Cropped puzzle cloud has", puzzle_cloud.size(), "points")
for name, pc in tray_clouds.items():
    print(f"Tray crop '{name}' has {pc.size()} points")

station_context = station.GetMyContextFromRoot(diagram_context)

puzzle_color_image = station.GetOutputPort("camera_puzzle.rgb_image").Eval(station_context)
puzzle_depth_image = station.GetOutputPort("camera_puzzle.depth_image").Eval(station_context)
tray_color_image = station.GetOutputPort("camera_tray.rgb_image").Eval(station_context)
tray_depth_image = station.GetOutputPort("camera_tray.depth_image").Eval(station_context)

meshcat.SetObject(
    "debug/puzzle/full",
    full_puzzle_cloud,
    point_size=0.005,
    rgba=Rgba(0.0, 0.0, 1.0),
)
meshcat.SetObject(
    "debug/puzzle/cropped",
    puzzle_cloud,
    point_size=0.01,
    rgba=Rgba(1.0, 0.0, 0.0),
)
meshcat.SetObject(
    "debug/tray/full",
    full_tray_cloud,
    point_size=0.005,
    rgba=Rgba(0.7, 0.7, 0.7),
)
for name, pc in tray_clouds.items():
    meshcat.SetObject(
        f"debug/tray/{name}",
        pc,
        point_size=0.01,
        rgba=Rgba(0.0, 1.0, 0.0),
    )

def _reshape_color_image(image):
    data = np.array(image.data, copy=False).reshape(
        image.height(), image.width(), -1
    )
    return data[..., :3]

def _reshape_depth_image(image):
    depth = np.array(image.data, copy=False).reshape(
        image.height(), image.width()
    )
    return np.ma.masked_invalid(depth)

puzzle_color = _reshape_color_image(puzzle_color_image)
puzzle_depth = _reshape_depth_image(puzzle_depth_image)
tray_color = _reshape_color_image(tray_color_image)
tray_depth = _reshape_depth_image(tray_depth_image)

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

# def crop_table(cloud, z_min):
#     xyz = np.asarray(cloud.xyzs())
#     keep = xyz[2, :] > z_min
#     new_cloud = type(cloud)(new_size=int(keep.sum()), fields=cloud.fields())
#     new_cloud.mutable_xyzs()[:] = xyz[:, keep]
#     if cloud.has_rgbs():
#         new_cloud.mutable_rgbs()[:] = np.asarray(cloud.rgbs())[:, keep]
#     return new_cloud
#     # return cloud

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

#     xs = np.arange(x_min, x_max + resolution, resolution)
#     ys = np.arange(y_min, y_max + resolution, resolution)
#     hmap = np.full((len(ys), len(xs)), np.nan)

#     i = np.floor((x - x_min) / resolution).astype(int)
#     j = np.floor((y - y_min) / resolution).astype(int)
#     for u, v, zz in zip(i, j, z):
#         if np.isnan(hmap[v, u]):
#             hmap[v, u] = zz
#         else:
#             hmap[v, u] = max(hmap[v, u], zz)
#     return hmap, xs, ys

# def signed_distance_box(xx, yy, box_min, box_max):
#     # box_min, box_max: (x_min, y_min), (x_max, y_max)
#     dx = np.maximum(np.maximum(box_min[0] - xx, 0), xx - box_max[0])
#     dy = np.maximum(np.maximum(box_min[1] - yy, 0), yy - box_max[1])
#     outside = (dx > 0) | (dy > 0)
#     dist_out = np.hypot(dx, dy)
#     dist_in = -np.minimum(np.minimum(xx - box_min[0], box_max[0] - xx),
#                           np.minimum(yy - box_min[1], box_max[1] - yy))
#     return np.where(outside, dist_out, dist_in)

# corners = np.array([
#     upper_left_translation[:2],
#     upper_right_translation[:2],
#     lower_left_translation[:2],
#     lower_right_translation[:2],
# ])
# x_min, y_min = corners.min(axis=0)
# x_max, y_max = corners.max(axis=0)
# margin = 0.005

# box_min = (x_min - margin, y_min - margin)
# box_max = (x_max + margin, y_max + margin)

# def visualize_height_and_gradient(cloud, resolution=0.005, step=1,
#                                   box_min=box_min, box_max=box_max):
#     hmap, xs, ys = cloud_height_map(cloud, resolution)
#     if not np.isfinite(hmap).any():
#         print("Height map is empty"); return
#     mean_val = np.nanmean(hmap)
#     hmap = np.where(np.isfinite(hmap), hmap, mean_val)

#     Gy, Gx = np.gradient(hmap, resolution, resolution)  # d/dy, d/dx
#     xx, yy = np.meshgrid(xs, ys)
#     phi = signed_distance_box(xx, yy, box_min, box_max)
#     phiy, phix = np.gradient(phi, resolution, resolution)

#     # Choose direction: outside -> inward to box; inside -> downhill height
#     dir_x = np.where(phi > 0, -phix, -Gx)
#     dir_y = np.where(phi > 0, -phiy, -Gy)
#     mag = np.hypot(dir_x, dir_y) + 1e-9
#     dir_x /= mag; dir_y /= mag

#     arrow_scale = 0.01
#     dx_draw, dy_draw = arrow_scale * dir_x, arrow_scale * dir_y

#     plt.figure(figsize=(8, 6))
#     plt.imshow(hmap, origin="lower",
#                extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="plasma")
#     plt.colorbar(label="Height (m)")
#     plt.quiver(xx[::step, ::step], yy[::step, ::step],
#                dx_draw[::step, ::step], dy_draw[::step, ::step],
#                color="white", angles="xy", scale_units="xy", scale=1.0,
#                width=0.002, headwidth=4, headlength=6, headaxislength=5,
#                pivot="mid")
#     plt.xlabel("X (m)"); plt.ylabel("Y (m)")
#     plt.title("Height map with inward/outward push vectors")
#     plt.tight_layout(); plt.show()

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

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(100)
