from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    InverseKinematics,
    RotationMatrix,
    Solve,
    RigidTransform,
)

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)
from pathlib import Path
import numpy as np
from controller import Controller



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


# welded puzzle frame translation
table_top_z = -0.05 + 0.025
puzzle_center_x = 0.0
puzzle_center_y = -0.60
puzzle_center_z = table_top_z
puzzle_offset = 0.01

upper_right_translation = (
    puzzle_center_x + puzzle_offset,
    puzzle_center_y + puzzle_offset,
    puzzle_center_z,
)
upper_left_translation = (
    puzzle_center_x - puzzle_offset,
    puzzle_center_y + puzzle_offset,
    puzzle_center_z,
)
lower_left_translation = (
    puzzle_center_x - puzzle_offset,
    puzzle_center_y - puzzle_offset,
    puzzle_center_z,
)
lower_right_translation = (
    puzzle_center_x + puzzle_offset,
    puzzle_center_y - puzzle_offset,
    puzzle_center_z,
)
cross_translation = (puzzle_center_x + 0.04, puzzle_center_y, puzzle_center_z + 0.01)

# tray piece translations
trapezoid_translation = (-0.15, 0.55, table_top_z)
infinity_translation = (-0.15, 0.80, table_top_z)
my_piece_translation = (0.15, 0.75, table_top_z)
rectangle_translation = (0.0, 0.90, table_top_z)


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
"""
scenario = LoadScenario(data=scenario_string)
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)

plant = station.GetSubsystemByName("plant")
plant_context = plant.CreateDefaultContext()
ik = InverseKinematics(plant, plant_context)
q = ik.q()
q_initial = get_hardcoded_initial_gripper_pose(plant, plant_context, cross_translation)
plant.SetDefaultPositions(q_initial)

controller = builder.AddSystem(Controller(q_desired=q_initial))

builder.Connect(
    station.GetOutputPort("iiwa_state"),  # or similar state port
    controller.get_input_port(0),
)
builder.Connect(
    controller.get_output_port(0),
    station.GetInputPort("iiwa_actuation"),
)

diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(50)
