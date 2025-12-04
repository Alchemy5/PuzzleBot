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

from src.missing_piece_estimation import (
    find_closest_z_center,
    find_z_centers,
    largest_region,
    cloud_similarity,
)

from src.motion_planning import run_ik, MotionController


def _format_vec(vec: tuple[float, float, float]) -> str:
    return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"

<<<<<<< HEAD:main.py
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

=======
>>>>>>> 5482ae3762e5cef108fb0023c56d027436110689:archive/varun/main.py

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

<<<<<<< HEAD:main.py
wsg_ctrl = builder.AddSystem(KeepWsgOpen(target_width=0.01))
builder.Connect(station.GetOutputPort("wsg_state"), wsg_ctrl.state_port)
builder.Connect(wsg_ctrl.get_output_port(0), station.GetInputPort("wsg_actuation"))

phi_ctrl = builder.AddSystem(PushAlongPhiController(plant))
builder.Connect(station.GetOutputPort("iiwa_state"), phi_ctrl.state_port)
builder.Connect(phi_ctrl.get_output_port(0), station.GetInputPort("iiwa_actuation"))

# grasp_pos = cross_translation + np.array([0, 0.04, 0.03])
# q_grasp = ik_to_hover_pose(plant, plant_context, grasp_pos)

# plant.SetDefaultPositions(q_grasp)
=======
# controller = builder.AddSystem(DepthController(plant))
motion_controller = builder.AddSystem(MotionController(output_size=7))
builder.Connect(
    motion_controller.get_output_port(), station.GetInputPort("iiwa_actuation")
)

# builder.Connect(
#    station.GetOutputPort("camera_puzzle.depth_image"),
#    controller.depth_port,
# )
# builder.Connect(
#    station.GetOutputPort("iiwa_generalized_contact_forces"),
#    controller.contact_port,
# )

# builder.Connect(
#    controller.get_output_port(0),
#    station.GetInputPort("iiwa_actuation"),
# )
>>>>>>> 5482ae3762e5cef108fb0023c56d027436110689:archive/varun/main.py

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

####### Perception and Motion Planning ########

puzzle_points = puzzle_cloud.xyzs().T

tray_piece_tight_clouds = {}  # dict to map name of piece to refined positive clouds
for piece in tray_clouds:
    cloud = tray_clouds[piece]
    points = cloud.xyzs().T
    center1, center2 = find_z_centers(puzzle_points)

    min_center = min(center1, center2)
    max_center = max(center1, center2)

    # we want max center now
    positive_space_points = []
    for point in points:
        closest_center = find_closest_z_center(point, min_center, max_center)
        if closest_center == max_center:
            positive_space_points.append(point)

    pos = largest_region(positive_space_points)

    cloud_pos = PointCloud(new_size=pos.shape[0])
    cloud_pos.mutable_xyzs()[:] = pos.T
    meshcat.SetObject(
        piece,
        cloud_pos,
        point_size=0.01,
        rgba=Rgba(0.0, 1.0, 0.0),
    )

    tray_piece_tight_clouds[piece] = pos


# Identify negative space
center1, center2 = find_z_centers(puzzle_points)
min_center = min(center1, center2)  # corresponds to negative space
max_center = max(center1, center2)  # corresponds to boundary puzzle pieces

negative_space_points = []
for point in puzzle_points:
    closest_center = find_closest_z_center(point, min_center, max_center)
    if closest_center == min_center:
        negative_space_points.append(point)

# now choose largest continuous region for these negative space points

neg_pts = largest_region(negative_space_points)

cloud_neg = PointCloud(new_size=neg_pts.shape[0])
cloud_neg.mutable_xyzs()[:] = neg_pts.T
meshcat.SetObject(
    "negative_space",
    cloud_neg,
    point_size=0.01,
    rgba=Rgba(0.0, 1.0, 0.0),
)

scores = {}
# Compute similarity scores between tray pieces and missing piece
for piece, pos_pts in tray_piece_tight_clouds.items():
    print(f"######## {piece} and missing piece (cross) similarity score ########")
    score, newB, R, t = cloud_similarity(neg_pts, pos_pts)
    print(f"Score: {score}")
    scores[piece] = {"score": score, "rotation": R, "translation": t, "cloud": pos_pts}
    if piece == "cross":
        cloud_translated = PointCloud(new_size=newB.shape[0])
        cloud_translated.mutable_xyzs()[:] = newB.T
        meshcat.SetObject(
            f"similarity - cross - {piece}",
            cloud_translated,
            point_size=0.01,
            rgba=Rgba(1.0, 0.0, 0.0),  # bright red to stand out
        )
        print(f"Rotation Matrix: {R}")
        print(f"Translation: {t}")
best_piece, best_entry = max(scores.items(), key=lambda item: item[1]["score"])
cloud = best_entry["cloud"]
piece_location = cloud.mean(axis=0)

################## Inverse Kinematics to move arm ##################
# given R and t how to move arm
plant_context = plant.GetMyContextFromRoot(diagram_context)

iiwa_model = plant.GetModelInstanceByName("iiwa")
q_init = plant.GetPositions(plant_context, iiwa_model)

q_grasp = run_ik(plant, plant_context, piece_location, q_init)[:7]
# TODO: create joint space trajectory from initial to q_grasp

T = 3.0
q_traj = PiecewisePolynomial.FirstOrderHold(
    [0.0, T], np.column_stack((q_init, q_grasp))
)
print
motion_controller.set_trajectory(q_traj)

<<<<<<< HEAD:main.py
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

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(100)
=======
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(10)
>>>>>>> 5482ae3762e5cef108fb0023c56d027436110689:archive/varun/main.py
