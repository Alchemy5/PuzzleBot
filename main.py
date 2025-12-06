"""
Imports
"""
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
    Sphere
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
from controller import Controller, DepthController, KeepWsgOpen

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


"""
Start meshcat and establish directory structure.
"""
meshcat = StartMeshcat()
def _format_vec(vec: tuple[float, float, float]) -> str:
    return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"
repo_root = Path("/Users/jity/Desktop/6.4210/PuzzleBot")
assets_dir = repo_root / "assets"

"""
Set up environment string for hardware station.
"""

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
tray_pcd_sys = pcd_systems["camera_tray"]

puzzle_pcd_port = puzzle_pcd_sys.point_cloud_output_port()
tray_pcd_port = tray_pcd_sys.point_cloud_output_port()

builder.ExportOutput(puzzle_pcd_port, "puzzle.point_cloud")
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
# wsg_ctrl = builder.AddSystem(KeepWsgOpen(target_width=0.01))
# builder.Connect(station.GetOutputPort("wsg_state"), wsg_ctrl.state_port)
# builder.Connect(wsg_ctrl.get_output_port(0), station.GetInputPort("wsg_actuation"))

diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context) 
plant_context = plant.GetMyContextFromRoot(context)

"""
Run perception functions to get point cloud data.
"""
# full_puzzle_cloud = get_puzzle_pointcloud(diagram, context)
# full_tray_cloud = get_tray_pointcloud(diagram, context)

# puzzle_cloud, tray_clouds = get_puzzle_and_tray_pointclouds(
#     diagram,
#     context,
#     puzzle_center=puzzle_center,
#     tray_translations=tray_translations,
# )

"""
Run analysis on perception data and compute target pose, etc.
"""


# puzzle_points = puzzle_cloud.xyzs().T

# tray_piece_tight_clouds = {}  # dict to map name of piece to refined positive clouds
# for piece in tray_clouds:
#     cloud = tray_clouds[piece]
#     points = cloud.xyzs().T
#     center1, center2 = find_z_centers(puzzle_points)

#     min_center = min(center1, center2)
#     max_center = max(center1, center2)

#     # we want max center now
#     positive_space_points = []
#     for point in points:
#         closest_center = find_closest_z_center(point, min_center, max_center)
#         if closest_center == max_center:
#             positive_space_points.append(point)

#     pos = largest_region(positive_space_points)

#     cloud_pos = PointCloud(new_size=pos.shape[0])
#     cloud_pos.mutable_xyzs()[:] = pos.T
#     meshcat.SetObject(
#         piece,
#         cloud_pos,
#         point_size=0.01,
#         rgba=Rgba(0.0, 1.0, 0.0),
#     )

#     tray_piece_tight_clouds[piece] = pos


# Identify negative space
# center1, center2 = find_z_centers(puzzle_points)
# min_center = min(center1, center2)  # corresponds to negative space
# max_center = max(center1, center2)  # corresponds to boundary puzzle pieces

# negative_space_points = []
# for point in puzzle_points:
#     closest_center = find_closest_z_center(point, min_center, max_center)
#     if closest_center == min_center:
#         negative_space_points.append(point)

# now choose largest continuous region for these negative space points

# neg_pts = largest_region(negative_space_points)

# cloud_neg = PointCloud(new_size=neg_pts.shape[0])

# cloud_neg_avg = neg_pts.mean(axis=0)
# cloud_neg.mutable_xyzs()[:] = neg_pts.T
# meshcat.SetObject(
#     "negative_space",
#     cloud_neg,
#     point_size=0.01,
#     rgba=Rgba(0.0, 1.0, 0.0),
# )

# scores = {}
# # Compute similarity scores between tray pieces and missing piece
# for piece, pos_pts in tray_piece_tight_clouds.items():
#     print(f"######## {piece} and missing piece (cross) similarity score ########")
#     score, newB, R, t = cloud_similarity(neg_pts, pos_pts)
#     print(f"Score: {score}")
#     scores[piece] = {"score": score, "rotation": R, "translation": t, "cloud": pos_pts}
#     if piece == "cross":
#         cloud_translated = PointCloud(new_size=newB.shape[0])
#         cloud_translated.mutable_xyzs()[:] = newB.T
#         meshcat.SetObject(
#             f"similarity - cross - {piece}",
#             cloud_translated,
#             point_size=0.01,
#             rgba=Rgba(1.0, 0.0, 0.0),  # bright red to stand out
#         )
#         print(f"Rotation Matrix: {R}")
#         print(f"Translation: {t}")
# best_piece, best_entry = max(scores.items(), key=lambda item: item[1]["score"])
# cloud = best_entry["cloud"]
# piece_location = cloud.mean(axis=0)

import numpy as np
from pydrake.all import InverseKinematics, Solve, PiecewisePolynomial, RotationMatrix

class LinearInterpolationPlanner:
    def __init__(self, plant, gripper_frame_name="body"):
        self.plant = plant
        self.gripper_frame = plant.GetFrameByName(gripper_frame_name)

    def _solve_ik(self, X_WG_target, context, orientation_tolerance=0.001, pos_tol=0.001):
        """
        Solve IK for a target pose.
        
        Args:
            X_WG_target: Target RigidTransform for gripper in world frame
            context: Plant context
            orientation_tolerance: Tolerance for orientation in radians
            pos_tol: Tolerance for position in meters (default 1cm)
        """
        ik = InverseKinematics(self.plant, context)
        q = ik.q()

        # Position constraint: point on gripper at target position
        p_W = X_WG_target.translation()
        ik.AddPositionConstraint(
            self.gripper_frame, np.array([0, 0.1, 0]),
            self.plant.world_frame(),
            p_W - pos_tol, p_W + pos_tol
        )

        R_WG_des = RotationMatrix.MakeXRotation(-np.pi / 2)  # flip around X so z points down
        ik.AddOrientationConstraint(
            self.gripper_frame, RotationMatrix(),
            self.plant.world_frame(), R_WG_des,
            orientation_tolerance
        )

        # Use current configuration as initial guess
        q_current = self.plant.GetPositions(context)
        ik.prog().SetInitialGuess(q, q_current)
        
        result = Solve(ik.prog())
        if not result.is_success():
            print(f"IK failed for target pose {p_W}")
            print(f"Solver: {result.get_solver_id().name()}")
            print(f"Current config: {q_current}")
            raise RuntimeError("IK failed for target pose")
        return result.GetSolution(q)
    
    def _solve_start_ik(self, X_WG_target, context, orientation_tolerance=0.001, pos_tol=0.001):
        """
        Solve IK for a target pose.
        
        Args:
            X_WG_target: Target RigidTransform for gripper in world frame
            context: Plant context
            orientation_tolerance: Tolerance for orientation in radians
            pos_tol: Tolerance for position in meters (default 1cm)
        """
        ik = InverseKinematics(self.plant, context)
        q = ik.q()

        # Position constraint: point on gripper at target position
        p_W = X_WG_target.translation()
        ik.AddPositionConstraint(
            self.gripper_frame, np.array([0, 0.1, 0]),
            self.plant.world_frame(),
            p_W - pos_tol, p_W + pos_tol
        )

        # Use current configuration as initial guess
        q_current = self.plant.GetPositions(context)
        ik.prog().SetInitialGuess(q, q_current)
        
        result = Solve(ik.prog())
        if not result.is_success():
            print(f"IK failed for target pose {p_W}")
            print(f"Solver: {result.get_solver_id().name()}")
            print(f"Current config: {q_current}")
            raise RuntimeError("IK failed for target pose")
        return result.GetSolution(q)

    def plan(self, X_WStart, X_WGoal, context, duration=2.0):
        """
        Plan a linear interpolation trajectory between two poses.
        """
        print("Solving IK for start pose...")
        q_start = plant.GetPositions(plant_context)
        print(f"✓ Start pose solved: {q_start}")
        
        print("Solving IK for goal pose...")
        q_goal = self._solve_ik(X_WGoal, context)
        print(f"✓ Goal pose solved: {q_goal}")
        
        times = [0.0, duration]
        positions = np.column_stack([q_start, q_goal])
        return PiecewisePolynomial.FirstOrderHold(times, positions)


sim = Simulator(diagram, context)
sim.Initialize()
sim.set_target_realtime_rate(0.5)
plant_context = plant.GetMyContextFromRoot(sim.get_mutable_context())

wsg = plant.GetModelInstanceByName("wsg")
iiwa = plant.GetModelInstanceByName("iiwa")

q_wsg = plant.GetPositions(plant_context, wsg)  # [q_l, q_r]
q_wsg[0] = -0.02
q_wsg[1] =  0.02
plant.SetPositions(plant_context, wsg, q_wsg)
open_wsg = np.array(q_wsg)  # save the open pose

sim.AdvanceTo(sim.get_context().get_time() + 0.5)
# diagram.ForcedPublish(context)
# meshcat.Flush()
print("WSG forced open:", q_wsg)

gripper_frame = plant.GetFrameByName("body")
X_WStart = plant.CalcRelativeTransform(
    plant_context, plant.world_frame(), gripper_frame
)
X_WGoal = RigidTransform(cross_translation + np.array([0, 0.033, 0.04]))
# Instantiate once
planner = LinearInterpolationPlanner(plant, gripper_frame_name="body")

# Plan from your poses (using the existing plant_context)
traj = planner.plan(X_WStart, X_WGoal, plant_context, duration=3.0)

t0 = sim.get_context().get_time()
dt = 0.01
t_final = t0 + traj.end_time()
t = t0
while t < t_final:
    tau = traj.start_time() + (t - t0)  # trajectory time
    q = traj.value(tau).flatten()
    plant.SetPositions(plant_context, iiwa, q[:7])
    # keep gripper open during approach (model-instance order-safe)
    plant.SetPositions(plant_context, wsg, open_wsg)
    plant.SetVelocities(plant_context, iiwa, np.zeros(7))
    plant.SetVelocities(plant_context, wsg, np.zeros(2))
    t = min(t + dt, t_final)
    sim.AdvanceTo(t)

# descend
X_WStart = plant.CalcRelativeTransform(
    plant_context, plant.world_frame(), gripper_frame
)
X_WGoal = RigidTransform(cross_translation + np.array([0, 0.033, 0.02]))
traj = planner.plan(X_WStart, X_WGoal, plant_context, duration=3.0)
t0 = sim.get_context().get_time()
dt = 0.01
t_final = t0 + traj.end_time()
t = t0
while t < t_final:
    tau = traj.start_time() + (t - t0)  # trajectory time
    q = traj.value(tau).flatten()
    plant.SetPositions(plant_context, iiwa, q[:7])
    # keep gripper open during approach (model-instance order-safe)
    plant.SetPositions(plant_context, wsg, open_wsg)
    plant.SetVelocities(plant_context, iiwa, np.zeros(7))
    plant.SetVelocities(plant_context, wsg, np.zeros(2))
    t = min(t + dt, t_final)
    sim.AdvanceTo(t)

q_wsg = plant.GetPositions(plant_context, wsg)  # [q_l, q_r]
q_wsg[0] = -0.009 # -0.01 - 0.00001 / 2
q_wsg[1] =  0 # 0.01 + 0.00001 / 2
closed_wsg = np.array(q_wsg)
q_hold = plant.GetPositions(plant_context, iiwa)
# plant.SetPositions(plant_context, wsg, q_wsg)
t0 = sim.get_context().get_time()
t_final = t0 + 0.5
t = t0
while t < t_final:
    alpha = (t - t0) / (t_final - t0)
    q_wsg = open_wsg * (1 - alpha) + closed_wsg * alpha
    plant.SetPositions(plant_context, iiwa, q_hold)
    plant.SetPositions(plant_context, wsg, q_wsg)
    plant.SetVelocities(plant_context, iiwa, np.zeros(7))
    plant.SetVelocities(plant_context, wsg, np.zeros(2))
    sim.AdvanceTo(min(t + dt, t_final))
    t = sim.get_context().get_time()

# diagram.ForcedPublish(context)
# meshcat.Flush()
print("WSG forced closed:", q_wsg)

X_WStart = plant.CalcRelativeTransform(
    plant_context, plant.world_frame(), gripper_frame
)
X_WGoal = RigidTransform(cross_translation + np.array([0, 0.033, 0.06]))

# Plan from your poses (using the existing plant_context)
traj = planner.plan(X_WStart, X_WGoal, plant_context, duration=3.0)

t0 = sim.get_context().get_time()
dt = 0.01
t_final = t0 + traj.end_time()
t = t0
while t < t_final:
    tau = traj.start_time() + (t - t0)
    q = traj.value(tau).flatten()
    plant.SetPositions(plant_context, iiwa, q[:7])
    # keep gripper closed during lift
    plant.SetPositions(plant_context, wsg, closed_wsg)
    plant.SetVelocities(plant_context, iiwa, np.zeros(7))
    plant.SetVelocities(plant_context, wsg, np.zeros(2))
    t = min(t + dt, t_final)
    sim.AdvanceTo(t)

# Play it in MeshCat
# PublishPositionTrajectory(traj, context, plant, visualizer)
