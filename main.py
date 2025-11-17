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
from pydrake.perception import PointCloud

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds,
)
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import trimesh
from controller import Controller, DepthController

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

from src.missing_piece_estimation import (
    find_closest_z_center,
    find_z_centers,
    largest_region,
)


def _format_vec(vec: tuple[float, float, float]) -> str:
    return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"


def get_hardcoded_initial_gripper_pose(plant, plant_context, cross_translation):
    # desired gripper pose, hover directly above the cross piece
    hover_height = 0.20
    p_WG_des = np.array(cross_translation) + np.array([0.0, 0.0, hover_height])

    R_WG_des = RotationMatrix.MakeXRotation(-np.pi / 2)
    X_WG_des = RigidTransform(R_WG_des, p_WG_des)

    W = plant.world_frame()
    wsg_model = plant.GetModelInstanceByName("wsg")
    G = plant.GetBodyByName("body", wsg_model).body_frame()

    ik.AddPositionConstraint(
        G,  # frameB
        [0.0, 0.0, 0.0],  # p_BQ
        W,  # frameA
        X_WG_des.translation() - 1e-3,  # p_AQ_lower
        X_WG_des.translation() - 1e-3,  # p_AQ_upper
    )

    ik.AddOrientationConstraint(
        W,  # frameAbar
        R_WG_des,  # R_AbarA
        G,  # frameBbar
        RotationMatrix(),  # R_BbarB
        1e-3,  # theta_bound
    )

    # small quadratic cost to keep solution well-behaved
    prog = ik.prog()
    prog.AddQuadraticErrorCost(np.eye(len(q)), np.zeros(len(q)), q)

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
- add_weld:
    parent: world
    child: puzzle_cross::cross_link
    X_PC:
        translation: {_format_vec(cross_translation)}
        rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_model:
    name: puzzle_camera
    file: "package://manipulation/camera_box.sdf"
- add_weld:
    parent: world
    child: puzzle_camera::base
    X_PC:
        translation: {_format_vec(camera_translation)}
        rotation: !Rpy {{ deg: [-160, 0, 0] }}

- add_model:
    name: tray_camera
    file: "package://manipulation/camera_box.sdf"
- add_weld:
    parent: world
    child: tray_camera::base
    X_PC:
        translation: {_format_vec(tray_camera_translation)}
        rotation: !Rpy {{ deg: [-150, 0, 0] }}

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

# plant_context = plant.CreateDefaultContext()
# ik = InverseKinematics(plant, plant_context)
# q = ik.q()
# q_initial = get_hardcoded_initial_gripper_pose(plant, plant_context, cross_translation)
# plant.SetDefaultPositions(q_initial)

# controller = builder.AddSystem(Controller(q_desired=q_initial))
controller = builder.AddSystem(DepthController(plant))

builder.Connect(
    station.GetOutputPort("camera_puzzle.depth_image"),
    controller.depth_port,
)
builder.Connect(
    station.GetOutputPort("iiwa_generalized_contact_forces"),
    controller.contact_port,
)
# builder.Connect(
#     station.GetOutputPort("iiwa_state"),  # or similar state port
#     controller.get_input_port(0),
# )
builder.Connect(
    controller.get_output_port(0),
    station.GetInputPort("iiwa_actuation"),
)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.ForcedPublish(diagram_context)

full_puzzle_cloud = get_puzzle_pointcloud(diagram, diagram_context)
full_tray_cloud = get_tray_pointcloud(diagram, diagram_context)

puzzle_cloud, tray_clouds = get_puzzle_and_tray_pointclouds(
    diagram,
    diagram_context,
    puzzle_center=puzzle_center,
    tray_translations=tray_translations,
)

####### TODO: Move following section to its own separate file ########
points = puzzle_cloud.xyzs().T
# Step 1: Identify negative space
center1, center2 = find_z_centers(points)
min_center = min(center1, center2)  # corresponds to negative space
max_center = max(center1, center2)  # corresponds to boundary puzzle pieces

negative_space_points = []
for point in points:
    closest_center = find_closest_z_center(point, min_center, max_center)
    if closest_center == min_center:
        negative_space_points.append(point)

# now choose largest continuous region for these negative space points

largest_negative_region = largest_region(negative_space_points)

neg_pts = np.asarray(largest_negative_region)
cloud_neg = PointCloud(new_size=neg_pts.shape[0])
cloud_neg.mutable_xyzs()[:] = neg_pts.T
meshcat.SetObject(
    "negative_space",
    cloud_neg,
    point_size=0.01,
    rgba=Rgba(0.0, 1.0, 0.0),
)
import pdb

pdb.set_trace()
######################################################################
print("Puzzle camera cloud has", full_puzzle_cloud.size(), "points")
print("Tray camera cloud has", full_tray_cloud.size(), "points")
print("Cropped puzzle cloud has", puzzle_cloud.size(), "points")
for name, pc in tray_clouds.items():
    print(f"Tray crop '{name}' has {pc.size()} points")

station_context = station.GetMyContextFromRoot(diagram_context)

puzzle_color_image = station.GetOutputPort("camera_puzzle.rgb_image").Eval(
    station_context
)
puzzle_depth_image = station.GetOutputPort("camera_puzzle.depth_image").Eval(
    station_context
)
tray_color_image = station.GetOutputPort("camera_tray.rgb_image").Eval(station_context)
tray_depth_image = station.GetOutputPort("camera_tray.depth_image").Eval(
    station_context
)

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
    data = np.array(image.data, copy=False).reshape(image.height(), image.width(), -1)
    return data[..., :3]


def _reshape_depth_image(image):
    depth = np.array(image.data, copy=False).reshape(image.height(), image.width())
    return np.ma.masked_invalid(depth)


puzzle_color = _reshape_color_image(puzzle_color_image)
puzzle_depth = _reshape_depth_image(puzzle_depth_image)
tray_color = _reshape_color_image(tray_color_image)
tray_depth = _reshape_depth_image(tray_depth_image)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(puzzle_color)
axes[0, 0].set_title("Puzzle camera RGB")
axes[0, 0].axis("off")

im = axes[0, 1].imshow(puzzle_depth, cmap="magma")
axes[0, 1].set_title("Puzzle camera depth")
axes[0, 1].axis("off")
fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

axes[1, 0].imshow(tray_color)
axes[1, 0].set_title("Tray camera RGB")
axes[1, 0].axis("off")

im = axes[1, 1].imshow(tray_depth, cmap="magma")
axes[1, 1].set_title("Tray camera depth")
axes[1, 1].axis("off")
fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(50)
