import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    Parser,
    RigidTransform,
    Solve,
    Sphere,
    StartMeshcat,
)

from pydrake.systems.framework import LeafSystem, BasicVector


# given goal destination in world coordinates, figure out q of robotic arm to get there
def run_ik(plant, plant_context, p_W_des, q0):

    ik = InverseKinematics(plant, plant_context)
    q = ik.q()[:7]

    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("body")

    ik.AddPositionConstraint(
        frameB=ee_frame,
        p_BQ=[0.0, 0.1, 0.0],
        frameA=world_frame,
        p_AQ_lower=p_W_des - 0.001,
        p_AQ_upper=p_W_des + 0.001,
    )

    prog = ik.prog()
    prog.SetInitialGuess(q, q0)
    result = Solve(prog)

    if not result.is_success():
        raise RuntimeError("IK failed")

    q_star = result.GetSolution(q)
    return q_star


# create simple controller class
class MotionController(LeafSystem):
    def __init__(self, output_size):
        LeafSystem.__init__(self)
        self._output_size = output_size
        self._trajectory = None

        self.DeclareVectorOutputPort(
            "position", BasicVector(output_size), self._calc_output
        )

    def set_trajectory(self, trajectory):
        self._trajectory = trajectory

    def _calc_output(self, context, output):
        if self._trajectory is None:
            output.SetFromVector(np.zeros(self._output_size))
        else:
            t = context.get_time()
            output.SetFromVector(self._trajectory.value(t).flatten())
