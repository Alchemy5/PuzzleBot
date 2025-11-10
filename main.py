from pydrake.all import RigidTransform, RollPitchYaw, StartMeshcat
from manipulation.letter_generation import create_sdf_asset_from_letter
import time


import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Concatenate,
    Context,
    Diagram,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    Rgba,
    RigidTransform,
    RobotDiagram,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    Trajectory,
    TrajectorySource,
)

from manipulation import running_as_notebook
from manipulation.exercises.grader import Grader
from manipulation.icp import IterativeClosestPoint
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import (
    AddPointClouds,
    LoadScenario,
    MakeHardwareStation,
    RobotDiagram,
)
from manipulation.utils import RenderDiagram



# initial pose / target pose / grasp pose functions

# motion planning function

# controller class

# inverse kinematics function (spatial position -> q)

# main function (that builds diagram + connects everything together)

# table sdf / five puzzle sdfs / almost complete puzzle sdf