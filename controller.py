import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    Context,
    AbstractValue,
    ImageDepth32F,
    RotationMatrix,
)
from puzzle_config import (
    upper_left_translation,
    upper_right_translation,
    lower_left_translation,
    lower_right_translation,
    puzzle_center_x,
    puzzle_center_y,
)

import numpy as np
from pydrake.all import LeafSystem, BasicVector, RotationMatrix, JacobianWrtVariable


def cloud_height_map(cloud, resolution=0.005):
    xyz = np.asarray(cloud.xyzs())
    x, y, z = xyz
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    half_x = resolution * np.ceil(
        max(puzzle_center_x - x_min, x_max - puzzle_center_x) / resolution
    )
    half_y = resolution * np.ceil(
        max(puzzle_center_y - y_min, y_max - puzzle_center_y) / resolution
    )

    n_x = int(np.ceil(2 * half_x / resolution))
    if n_x % 2 == 0:
        n_x += 1  # force an odd count so the center sample exists
    n_y = int(np.ceil(2 * half_y / resolution))
    if n_y % 2 == 0:
        n_y += 1

    xs = puzzle_center_x + resolution * np.arange(-(n_x // 2), n_x // 2 + 1)
    ys = puzzle_center_y + resolution * np.arange(-(n_y // 2), n_y // 2 + 1)

    hmap = np.full((len(ys), len(xs)), np.nan)

    i = np.floor((x - xs[0]) / resolution).astype(int)
    j = np.floor((y - ys[0]) / resolution).astype(int)
    valid = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
    for u, v, zz in zip(i[valid], j[valid], z[valid]):
        hmap[v, u] = zz if np.isnan(hmap[v, u]) else max(hmap[v, u], zz)
    return hmap, xs, ys


def signed_distance_box(xx, yy, box_min, box_max):
    # box_min, box_max: (x_min, y_min), (x_max, y_max)
    dx = np.maximum(np.maximum(box_min[0] - xx, 0), xx - box_max[0])
    dy = np.maximum(np.maximum(box_min[1] - yy, 0), yy - box_max[1])
    outside = (dx > 0) | (dy > 0)
    dist_out = np.hypot(dx, dy)
    dist_in = -np.minimum(
        np.minimum(xx - box_min[0], box_max[0] - xx),
        np.minimum(yy - box_min[1], box_max[1] - yy),
    )
    return np.where(outside, dist_out, dist_in)


corners = np.array(
    [
        upper_left_translation[:2],
        upper_right_translation[:2],
        lower_left_translation[:2],
        lower_right_translation[:2],
    ]
)
x_min, y_min = corners.min(axis=0)
x_max, y_max = corners.max(axis=0)
margin = 0.005

box_min = (x_min - margin, y_min - margin)
box_max = (x_max + margin, y_max + margin)


class Controller(LeafSystem):
    """PID controller for the IIWA robot"""

    def __init__(
        self, plant, iiwa, desired_z=0.12, resolution=0.005, step_gain=0.02
    ) -> None:
        LeafSystem.__init__(self)

        self.state_port = self.DeclareVectorInputPort("iiwa_state", 14)
        self.output_port = self.DeclareVectorOutputPort(
            "iiwa_torque", 7, self.ComputeTorque
        )
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.kp = 300
        self.kd = 200
        self.ki = 100
        self.qdot_desired = np.zeros(7)
        self.integral_error = np.zeros(7)
        self.iiwa = iiwa
        self.qs = None
        self.prev_time = 0.0
        self.idx = 0
        self.wsg_ctrl = None

        self.desired_z = desired_z
        self.step_gain = step_gain
        self.resolution = resolution

        self.Kp_pos, self.Kd_pos = 200, 20
        self.Kp_rot, self.Kd_rot = 200, 20

        self.wsg = plant.GetModelInstanceByName("wsg")
        self.gripper_body = plant.GetBodyByName("body", self.wsg)
        self.movement = False

    def _lookup_dir(self, x, y):
        i = int(np.clip(round((x - self.xs[0]) / self.resolution), 0, len(self.xs) - 1))
        j = int(np.clip(round((y - self.ys[0]) / self.resolution), 0, len(self.ys) - 1))
        return np.array([self.dir_x[j, i], self.dir_y[j, i]])

    def set_qs(self, qs):
        self.qs = qs

    def set_wsg_ctrl(self, wsg_ctrl):
        self.wsg_ctrl = wsg_ctrl

    def initialize_field(self, cloud):
        # Precompute vector field on the height grid
        hmap, xs, ys = cloud_height_map(cloud, self.resolution)
        hmap = np.where(np.isfinite(hmap), hmap, np.nanmean(hmap))
        hmap = np.pad(hmap, 1, mode="edge")
        hmap = (
            hmap[:-2, :-2]
            + hmap[:-2, 1:-1]
            + hmap[:-2, 2:]
            + hmap[1:-1, :-2]
            + hmap[1:-1, 1:-1]
            + hmap[1:-1, 2:]
            + hmap[2:, :-2]
            + hmap[2:, 1:-1]
            + hmap[2:, 2:]
        ) / 9.0

        Gy, Gx = np.gradient(hmap, self.resolution, self.resolution)

        xx, yy = np.meshgrid(xs, ys)
        phi = signed_distance_box(xx, yy, box_min, box_max)
        phiy, phix = np.gradient(phi, self.resolution, self.resolution)

        self.dir_x = np.where(phi > 0, -phix, -Gx)
        self.dir_y = np.where(phi > 0, -phiy, -Gy)
        self.xs, self.ys = xs, ys

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        if self.qs is None:  # check if any waypoints
            raise RuntimeError("initialize qs first")

        # Compute current state of iiwa q and qdot
        iiwa_state = self.state_port.Eval(context)

        q = iiwa_state[:7]
        qdot = iiwa_state[7:]

        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        self.plant.SetVelocities(self.plant_context, self.iiwa, qdot)

        q_des = self.qs[self.idx]

        # if we’re close enough, advance to the next waypoint
        err_norm = np.linalg.norm(q_des - q)
        print(self.idx, err_norm)

        # Note: when it's navigating from 0 to 1 stays open but as soon as at 1 gripper closes now makes sense

        if err_norm < 0.01 and self.idx < len(self.qs) - 1:
            if self.idx == 1:
                self.wsg_ctrl.set_target(0)  # close the gripper completely
            self.idx += 1
            q_des = self.qs[self.idx]
        elif (
            err_norm < 6.169018046688123e-02
            and self.idx
            == len(self.qs) - 1  # activate nudging when finally hit final waypoint
        ) or self.movement:  # static latch
            X_WG = self.plant.CalcRelativeTransform(
                self.plant_context,
                self.plant.world_frame(),
                self.gripper_body.body_frame(),
            )
            p_WG = X_WG.translation() - np.array([0, 0.030, 0])
            V_WG = self.plant.EvalBodySpatialVelocityInWorld(
                self.plant_context, self.gripper_body
            )

            dxy = self._lookup_dir(p_WG[0], p_WG[1])
            p_goal = np.array(
                [
                    p_WG[0] + self.step_gain * dxy[0],
                    p_WG[1] + self.step_gain * dxy[1],
                    self.desired_z,
                ]
            )

            e_p = p_goal - p_WG
            v_p = V_WG.translational()

            R_des = RotationMatrix.MakeXRotation(-np.pi / 2)
            R_err = R_des.multiply(X_WG.rotation().transpose())
            aa = R_err.ToAngleAxis()
            e_R = aa.angle() * aa.axis()

            f_W = self.Kp_pos * e_p - self.Kd_pos * v_p
            m_W = self.Kp_rot * e_R - self.Kd_rot * V_WG.rotational()
            wrench_W = np.hstack((m_W, f_W))

            J_WG = self.plant.CalcJacobianSpatialVelocity(
                self.plant_context,
                JacobianWrtVariable.kV,
                self.gripper_body.body_frame(),
                [0, 0, 0],
                self.plant.world_frame(),
                self.plant.world_frame(),
            )
            tau_full = J_WG.T @ wrench_W
            tau_full -= self.plant.CalcGravityGeneralizedForces(self.plant_context)
            output.set_value(tau_full[:7])

            self.movement = True

            print("tau is ", np.linalg.norm(f_W[:7]))
            if np.linalg.norm(f_W[:7]) < 0.01:
                self.wsg_ctrl.set_target(0.06)

            return

        current_time = context.get_time()
        dt = current_time - self.prev_time

        # Compute position and velocity errors (same as PD controller)
        position_error = q_des - q
        velocity_error = self.qdot_desired - qdot

        # Update integral error
        if dt > 0:  # Avoid division by zero on first call
            self.integral_error += dt * position_error

        torque = (
            self.kp * position_error
            + self.kd * velocity_error
            + self.ki * self.integral_error
        )
        tau_g_full = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        self.prev_time = current_time

        output.set_value(torque - tau_g_full[:7])


class WsgController(LeafSystem):
    def __init__(self, target_width=0.06, kp=400.0, kd=50.0):
        super().__init__()
        self.target = target_width / 2.0  # each finger
        self.kp = kp
        self.kd = kd

        self.state_port = self.DeclareVectorInputPort("wsg_state", 4)
        self.DeclareVectorOutputPort("wsg_actuation", 2, self.CalcTau)

    def set_target(self, width):
        self.target = width / 2.0

    def CalcTau(self, context, output):
        if self.target == 0:
            output.SetFromVector([3.4575, -3.4575])  # closes the gripper via forces

        else:
            x = self.state_port.Eval(context).ravel()
            q_l, q_r, v_l, v_r = x  # get inputs from input port
            qd = self.target  #
            tau_l = (
                self.kp * (-qd - q_l) - self.kd * v_l
            )  # pid control formula to find l and r forces
            tau_r = self.kp * (qd - q_r) - self.kd * v_r
            output.SetFromVector([tau_l, tau_r])
