from pydrake.all import (
    RigidTransform,
    Rgba,
    Sphere,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    Solve,
)
from manipulation.meshcat_utils import PublishPositionTrajectory

import numpy as np


def move_robot(X_WStart, X_WGoal, meshcat, plant, plant_context):
    meshcat.SetObject("start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1))
    meshcat.SetTransform("start", X_WStart)
    meshcat.SetObject("goal", Sphere(0.02), rgba=Rgba(0.1, 0.9, 0.1, 1))
    meshcat.SetTransform("goal", X_WGoal)

    num_q = plant.num_positions()
    q0 = plant.GetPositions(plant_context)
    gripper_frame = plant.GetFrameByName("body")

    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 10)
    prog = trajopt.get_mutable_prog()
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)

    trajopt.AddDurationConstraint(0.5, 5)

    # start constraint
    start_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WStart.translation(),
        X_WStart.translation(),
        gripper_frame,
        [0, 0.1, 0],
        plant_context,
    )

    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0, trajopt.control_points()[:, 0])

    goal_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WGoal.translation(),
        X_WGoal.translation(),
        gripper_frame,
        [0, 0.1, 0],
        plant_context,
    )

    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0, trajopt.control_points()[:, -1])

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed")
        print(result.get_solver_id().name())
    trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

    trajectory = trajopt.ReconstructTrajectory(result)
    return trajectory
