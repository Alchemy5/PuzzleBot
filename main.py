from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
)

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)
from pathlib import Path


def _format_vec(vec: tuple[float, float, float]) -> str:
    return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"


# Start meshcat for visualization
meshcat = StartMeshcat()
print("Click the link above to open Meshcat in your browser!")


repo_root = Path(__file__).resolve().parent
assets_dir = repo_root / "assets"

# assets for tray pieces
triangle_sdf_uri = (assets_dir / "green_triangle.sdf").resolve().as_uri()
my_piece_sdf_uri = (assets_dir / "my_piece.sdf").resolve().as_uri()
rectangle_sdf_uri = (assets_dir / "rectangle.sdf").resolve().as_uri()
trapezoid_sdf_uri = (assets_dir / "trapezoid.sdf").resolve().as_uri()

square_sdf = ""  # jity
semicircle_sdf = ""  # jity
rectangle_sdf = ""  # varun
star_sdf = ""  # varun

# assets for welded puzzle frame
corner_sdf_uri = (assets_dir / "puzzle_corner.sdf").resolve().as_uri()
cross_sdf_uri = (assets_dir / "puzzle_cross.sdf").resolve().as_uri()


table_top_z = -0.05 + 0.025  # matches table placement below
puzzle_center_x = 0.65
puzzle_center_y = 0.0
puzzle_center_z = table_top_z
puzzle_offset = 0.07

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
cross_translation = (puzzle_center_x, puzzle_center_y, puzzle_center_z)
triangle_translation = (0.45, -0.30, table_top_z)
my_piece_translation = (0.15, 0.65, table_top_z)
rectangle_translation = (0.30, -0.30, table_top_z)
trapezoid_translation = (-0.30, -1.30, table_top_z)

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
    name: custom_trapezoid
    file: "{trapezoid_sdf_uri}"
- add_weld:
    parent: world
    child: custom_trapezoid::trapezoid_link
    X_PC:
        translation: {_format_vec(trapezoid_translation)}
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
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(50)
