from manipulation import ConfigureParser
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    InverseKinematics,
    RotationMatrix,
    Solve,
    RigidTransform,
    Rgba,
    PiecewisePolynomial,
    TrajectorySource,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    AddMultibodyPlantSceneGraph,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    MinimumDistanceLowerBoundConstraint,
    BsplineTrajectory,
    Sphere,
)
from pydrake.perception import PointCloud
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds,
)
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import trimesh
from controller import Controller, WsgController

from puzzle_pointclouds import (
    get_puzzle_and_tray_pointclouds,
    get_puzzle_pointcloud,
    get_tray_pointcloud,
    get_full_puzzle_pointcloud,
)
import time
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
    full_camera_translation,
)

from src.missing_piece_estimation import (
    find_closest_z_center,
    find_z_centers,
    largest_region,
    cloud_similarity,
)
from controller_visualization_functions import (
    visualize_height_and_gradient,
    plot_point_cloud,
    crop_table,
)


meshcat = StartMeshcat()


def _format_vec(vec: tuple[float, float, float]) -> str:
    return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"


repo_root = Path("/Users/varun/robotics_final_project")
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

- add_model:
    name: full_puzzle_camera
    file: "package://manipulation/camera_box.sdf"
- add_weld:
    parent: world
    child: full_puzzle_camera::base
    X_PC:
        translation: {_format_vec(full_camera_translation)}
        rotation: !Rpy {{ deg: [-180, 0, 0] }}

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

  full_puzzle_camera:
    name: full_puzzle_camera
    depth: true
    X_PB:
        base_frame: full_puzzle_camera::base

"""

"""
Fully create simulation environment and build core diagram.
"""

meshcat.Delete()
scenario = LoadScenario(data=scenario_string)
station = MakeHardwareStation(scenario)
builder = DiagramBuilder()

station_sys = builder.AddSystem(station)

# add point clouds
pcd_systems = AddPointClouds(builder=builder, station=station_sys, scenario=scenario)
puzzle_pcd_sys = pcd_systems["camera_puzzle"]
full_puzzle_pcd_sys = pcd_systems["full_puzzle_camera"]
tray_pcd_sys = pcd_systems["camera_tray"]

puzzle_pcd_port = puzzle_pcd_sys.point_cloud_output_port()
full_puzzle_port = full_puzzle_pcd_sys.point_cloud_output_port()
tray_pcd_port = tray_pcd_sys.point_cloud_output_port()

builder.ExportOutput(puzzle_pcd_port, "puzzle.point_cloud")
builder.ExportOutput(full_puzzle_port, "puzzle.full_point_cloud")
builder.ExportOutput(tray_pcd_port, "tray.point_cloud")

scene_graph = station_sys.GetSubsystemByName("scene_graph")
plant = station_sys.GetSubsystemByName("plant")

visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    station.GetOutputPort("query_object"),
    meshcat,
    MeshcatVisualizerParams(role=Role.kIllustration),
)
collision_visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    station.GetOutputPort("query_object"),
    meshcat,
    MeshcatVisualizerParams(
        prefix="collision", role=Role.kProximity, visible_by_default=False
    ),
)

wsg = plant.GetModelInstanceByName("wsg")
iiwa = plant.GetModelInstanceByName("iiwa")
gripper_frame = plant.GetFrameByName("body")

wsg_ctrl = builder.AddSystem(WsgController(target_width=0.06))
builder.Connect(station.GetOutputPort("wsg_state"), wsg_ctrl.state_port)
builder.Connect(wsg_ctrl.get_output_port(0), station.GetInputPort("wsg_actuation"))

iiwa_ctrl = builder.AddSystem(Controller(plant, iiwa))
builder.Connect(station.GetOutputPort("iiwa_state"), iiwa_ctrl.state_port)
builder.Connect(iiwa_ctrl.get_output_port(0), station.GetInputPort("iiwa_actuation"))

diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.CreateDefaultContext()
diagram.ForcedPublish(context)

nudging_cloud = get_full_puzzle_pointcloud(diagram, context)

xyz = nudging_cloud.xyzs()
np.save("nudging_cloud.npy", xyz)
visualize_height_and_gradient(nudging_cloud)
