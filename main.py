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


# Start meshcat for visualization
meshcat = StartMeshcat()
print("Click the link above to open Meshcat in your browser!")

# pieces to choose from
triangle_sdf_path = Path("/Users/jity/Desktop/6.4210/PuzzleBot/assets/green_triangle.sdf")
triangle_sdf_uri = triangle_sdf_path.resolve().as_uri()

square_sdf = ""
semicircle_sdf = ""
hexagon_sdf = ""
star_sdf = ""

# puzzle pieces
upper_left_corner_sdf = "" # marik
lower_left_corner_sdf = "" # marik
upper_right_corner_sdf = "" # marik
lower_right_corner_sdf = "" # marik
cross_sdf = "" # marik


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
    file: package://manipulation/table.sdf
- add_weld:
    parent: world
    child: table::table_link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
- add_model:
    name: green_triangle_piece
    file: "{triangle_sdf_uri}"
"""
scenario = LoadScenario(data=scenario_string)
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(50)
