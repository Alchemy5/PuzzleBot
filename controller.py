import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    Context,
    AbstractValue,
    ImageDepth32F,
    RotationMatrix
)
from puzzle_config import upper_left_translation, upper_right_translation, lower_left_translation, lower_right_translation
class Controller(LeafSystem):
    """PID controller for the IIWA robot"""

    def __init__(self, plant, iiwa) -> None:
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
        self.q_desired = np.array([-1.57, 0.1, 0, -1.2, 0, 1.6, 0])
        self.qdot_desired = np.zeros(7)
        self.integral_error = np.zeros(7)
        self.iiwa = iiwa
        self.qs = None
        self.prev_time = 0.0
        self.idx = 0
        self.wsg_ctrl = None
        self.tick_control = [20, ]
    
    def set_q_desired(self, q_desired):
        self.q_desired = q_desired
    
    def set_qs(self, qs):
        self.qs = qs
    
    def set_wsg_ctrl(self, wsg_ctrl):
        self.wsg_ctrl = wsg_ctrl

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        if self.qs is None:
            raise RuntimeError("initialize qs first")
        # TODO: Extract state information (same as PD controller)
        iiwa_state = self.state_port.Eval(context)
        q = iiwa_state[:7]  # YOUR CODE HERE
        qdot = iiwa_state[7:]  # YOUR CODE HERE

        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        self.plant.SetVelocities(self.plant_context, self.iiwa, qdot)

        q_des = self.qs[self.idx]

        # if we’re close enough, advance to the next waypoint
        err_norm = np.linalg.norm(q_des - q)
        # print(self.idx, len(self.qs), err_norm)
        if err_norm < 0.01 and self.idx < len(self.qs) - 1:
            if self.idx == 1:
                self.wsg_ctrl.set_target(0)
            self.idx += 1
            q_des = self.qs[self.idx]
            

        current_time = context.get_time()
        dt = current_time - self.prev_time

        # TODO: Compute position and velocity errors (same as PD controller)
        position_error = q_des - q
        velocity_error = self.qdot_desired - qdot

        # TODO: Update integral error
        if dt > 0:  # Avoid division by zero on first call
            self.integral_error += dt * position_error

        # TODO: Compute PID control law
        # HINT: Combine all three terms: proportional + derivative + integral
        torque = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error
        tau_g_full = self.plant.CalcGravityGeneralizedForces(self.plant_context)

        # Update previous time for next iteration
        self.prev_time = current_time

        output.set_value(torque - tau_g_full[:7])

class DepthController(LeafSystem):
    def __init__(self, plant) -> None:
        super().__init__()

        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()

        self.alpha = 3.0
        self.beta = 20.0

        # input 1: depth image
        sample_depth = ImageDepth32F(640, 480)
        self.depth_port = self.DeclareAbstractInputPort(
            "depth_image", AbstractValue.Make(sample_depth)
        )

        # input 2: generalized contact forces on iiwa (from station)
        self.contact_port = self.DeclareVectorInputPort(
            "iiwa_contact_forces", 7
        )

        # output: joint torques
        self.DeclareVectorOutputPort(
            "iiwa_torque", 7, self.CalcTorque
        )

    def CalcTorque(self, context: Context, output: BasicVector) -> None:
        # gradient term - TODO
        depth_img: ImageDepth32F = self.depth_port.Eval(context)
        depth = np.array(depth_img.data, copy=False)[:, :, 0]

        # gravity term
        tau_g_full = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        tau_g = tau_g_full[:7]

        tau = tau_grad - tau_g
        output.set_value(tau)

class WaypointPDController(LeafSystem):
    """
    joint-space pd + gravity along a sequence of waypoints.

    torque = kp * (q_des - q) - kd * qdot - tau_g
    """

    def __init__(
        self,
        plant,
        iiwa_model,
        q_waypoints: np.ndarray,   # shape (N, 7)
    ) -> None:
        super().__init__()

        self.plant = plant
        self.iiwa = iiwa_model
        self.plant_context = plant.CreateDefaultContext()

        self.kp = 50
        self.kd = 10
        self.q_waypoints = np.asarray(q_waypoints)
        self.num_waypoints = self.q_waypoints.shape[0]

        self.joint_threshold = 0.02
        self.current_index = 0  # we’ll just mutate this (simple but fine here)

        # input: iiwa state [q(7); qdot(7)]
        self.state_port = self.DeclareVectorInputPort("iiwa_state", 14)

        # output: iiwa torque
        self.DeclareVectorOutputPort("iiwa_torque", 7, self.CalcTorque)

    def CalcTorque(self, context: Context, output: BasicVector) -> None:
        # 1) read state
        x = self.state_port.Eval(context)
        q = x[:7]
        qdot = x[7:]

        # 2) pick current waypoint
        q_des = self.q_waypoints[self.current_index][:7]

        # if we’re close enough, advance to the next waypoint
        err_norm = np.linalg.norm(q_des - q)
        if err_norm < self.joint_threshold and self.current_index < self.num_waypoints - 1:
            self.current_index += 1
            q_des = self.q_waypoints[self.current_index]

        # 3) pd term
        position_error = q_des - q
        velocity_error = -qdot   # desired qdot = 0

        tau_pd = self.kp * position_error + self.kd * velocity_error

        # 4) gravity compensation
        # update plant context with iiwa q (we don’t care about other models here)
        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        self.plant.SetVelocities(self.plant_context, self.iiwa, qdot)

        tau_g_full = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        # for iiwa + wsg, iiwa is first 7 dofs
        tau_g = tau_g_full[:7]

        # 5) final torque
        tau = tau_pd - tau_g
        output.set_value(tau)


import numpy as np
from pydrake.all import LeafSystem, BasicVector, JacobianWrtVariable
from puzzle_config import puzzle_center

class PushToCenterController(LeafSystem):
    def __init__(self, plant, desired_z=0.15):
        super().__init__()
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.desired_z = desired_z
        self.Kp_pos = 100
        self.Kd_pos = 20
        self.Kp_rot = 200
        self.Kd_rot = 20

        self.q_dim = plant.num_positions()
        self.v_dim = plant.num_velocities()
        self.iiwa_model = plant.GetModelInstanceByName("iiwa")
        self.wsg_model = plant.GetModelInstanceByName("wsg")
        self.gripper_body = plant.GetBodyByName("body", self.wsg_model)

        # Input: iiwa state (positions + velocities)
        self.state_port = self.DeclareVectorInputPort(
            "iiwa_state", 14
        )
        # Output: joint torques (7)
        self.DeclareVectorOutputPort("iiwa_torque", 7, self.CalcTorque)

    def CalcTorque(self, context, output):
        state = self.state_port.Eval(context)
        q = state[:7]
        v = state[7:]

        # Update plant context
        self.plant.SetPositions(self.context, self.iiwa_model, q)
        self.plant.SetVelocities(self.context, self.iiwa_model, v)

        R_des = RotationMatrix.MakeXRotation(-np.pi/2) 

        # Desired point: puzzle center XY, specified Z
        p_des_W = np.array([puzzle_center[0], puzzle_center[1], self.desired_z])

        # Current gripper pose/velocity
        X_WG = self.plant.CalcRelativeTransform(
            self.context, self.plant.world_frame(), self.gripper_body.body_frame()
        )
        p_WG = X_WG.translation()
        R_WG = X_WG.rotation()
        V_WG = self.plant.EvalBodySpatialVelocityInWorld(
            self.context, self.gripper_body
        )

        # Planar error/velocity (XY)
        e_p = p_des_W - p_WG
        v_p = V_WG.translational()
        
        R_err = R_des.multiply(R_WG.transpose())
        aa = R_err.ToAngleAxis()
        e_R = aa.angle() * aa.axis()  # 3-vector

        # Desired Cartesian force in world XY
        Kp_pos, Kd_pos = self.Kp_pos, self.Kd_pos
        Kp_rot, Kd_rot = self.Kp_rot, self.Kd_rot
        f_W = Kp_pos * e_p - Kd_pos * v_p
        m_W = Kp_rot * e_R - Kd_rot * V_WG.rotational()
        wrench_W = np.hstack((m_W, f_W))

        # Spatial Jacobian (angular + translational) at gripper origin
        J_WG = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            JacobianWrtVariable.kV,
            self.gripper_body.body_frame(),
            [0, 0, 0],
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

        tau_full = J_WG.T @ wrench_W  # size = num_velocities
        tau_g = self.plant.CalcGravityGeneralizedForces(self.context)
        tau_full -= tau_g

        # Output only the iiwa’s 7 torques
        output.set_value(tau_full[:7])
        # output.set_value(-tau_g[:7])

import numpy as np
from pydrake.all import LeafSystem, BasicVector

class WsgController(LeafSystem):
    def __init__(self, target_width=0.06, kp=400.0, kd=50.0):
        super().__init__()
        self.target = target_width / 2.0  # each finger
        self.kp = kp
        self.kd = kd
        # wsg_state: [q_l, q_r, v_l, v_r]
        self.state_port = self.DeclareVectorInputPort("wsg_state", 4)
        self.DeclareVectorOutputPort("wsg_actuation", 2, self.CalcTau)

    def set_target(self, width):
        self.target = width / 2.0

    def CalcTau(self, context, output):
        if self.target == 0:
            output.SetFromVector([15, -15])
        else:
            x = self.state_port.Eval(context).ravel()
            q_l, q_r, v_l, v_r = x
            qd = self.target
            tau_l = self.kp * (-qd - q_l) - self.kd * v_l
            tau_r = self.kp * (qd - q_r) - self.kd * v_r
            output.SetFromVector([tau_l, tau_r])

import numpy as np
from pydrake.all import LeafSystem, BasicVector, RotationMatrix, JacobianWrtVariable

def cloud_height_map(cloud, resolution=0.005):
    xyz = np.asarray(cloud.xyzs())  # (3, N)
    x, y, z = xyz
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    hmap = np.full((len(ys), len(xs)), np.nan)

    i = np.floor((x - x_min) / resolution).astype(int)
    j = np.floor((y - y_min) / resolution).astype(int)
    for u, v, zz in zip(i, j, z):
        if np.isnan(hmap[v, u]):
            hmap[v, u] = zz
        else:
            hmap[v, u] = max(hmap[v, u], zz)
    return hmap, xs, ys

def signed_distance_box(xx, yy, box_min, box_max):
    # box_min, box_max: (x_min, y_min), (x_max, y_max)
    dx = np.maximum(np.maximum(box_min[0] - xx, 0), xx - box_max[0])
    dy = np.maximum(np.maximum(box_min[1] - yy, 0), yy - box_max[1])
    outside = (dx > 0) | (dy > 0)
    dist_out = np.hypot(dx, dy)
    dist_in = -np.minimum(np.minimum(xx - box_min[0], box_max[0] - xx),
                          np.minimum(yy - box_min[1], box_max[1] - yy))
    return np.where(outside, dist_out, dist_in)

corners = np.array([
    upper_left_translation[:2],
    upper_right_translation[:2],
    lower_left_translation[:2],
    lower_right_translation[:2],
])
x_min, y_min = corners.min(axis=0)
x_max, y_max = corners.max(axis=0)
margin = 0.005

box_min = (x_min - margin, y_min - margin)
box_max = (x_max + margin, y_max + margin)

class PushAlongPhiController(LeafSystem):
    def __init__(self, plant, desired_z=0.12, resolution=0.005, step_gain=0.02):
        super().__init__()
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.desired_z = desired_z
        self.step_gain = step_gain
        self.resolution = resolution

        self.Kp_pos, self.Kd_pos = 200, 20
        self.Kp_rot, self.Kd_rot = 200, 20

        self.iiwa = plant.GetModelInstanceByName("iiwa")
        self.wsg = plant.GetModelInstanceByName("wsg")
        self.gripper_body = plant.GetBodyByName("body", self.wsg)

        self.initialized = False

        self.state_port = self.DeclareVectorInputPort("iiwa_state", 14)
        self.DeclareVectorOutputPort("iiwa_torque", 7, self.CalcTorque)

    def initialize_field(self, cloud):
        # Precompute vector field on the height grid
        hmap, xs, ys = cloud_height_map(cloud, self.resolution)
        hmap = np.where(np.isfinite(hmap), hmap, np.nanmean(hmap))
        Gy, Gx = np.gradient(hmap, self.resolution, self.resolution)

        xx, yy = np.meshgrid(xs, ys)
        phi = signed_distance_box(xx, yy, box_min, box_max)
        phiy, phix = np.gradient(phi, self.resolution, self.resolution)

        dir_x = np.where(phi > 0, -phix, -Gx)
        dir_y = np.where(phi > 0, -phiy, -Gy)
        mag = np.hypot(dir_x, dir_y) + 1e-9
        self.dir_x = dir_x / mag
        self.dir_y = dir_y / mag
        self.xs, self.ys = xs, ys

        self.initialized = True

    def _lookup_dir(self, x, y):
        i = int(np.clip(round((x - self.xs[0]) / self.resolution), 0, len(self.xs) - 1))
        j = int(np.clip(round((y - self.ys[0]) / self.resolution), 0, len(self.ys) - 1))
        return np.array([self.dir_x[j, i], self.dir_y[j, i]])

    def CalcTorque(self, context, output):
        if not self.initialized:
            raise RuntimeError("Call initialize_field() first")

        state = self.state_port.Eval(context)
        q, v = state[:7], state[7:]

        self.plant.SetPositions(self.context, self.iiwa, q)
        self.plant.SetVelocities(self.context, self.iiwa, v)

        X_WG = self.plant.CalcRelativeTransform(
            self.context, self.plant.world_frame(), self.gripper_body.body_frame()
        )
        p_WG = X_WG.translation()
        V_WG = self.plant.EvalBodySpatialVelocityInWorld(self.context, self.gripper_body)

        dxy = self._lookup_dir(p_WG[0], p_WG[1])
        p_goal = np.array([
            p_WG[0] + self.step_gain * dxy[0],
            p_WG[1] + self.step_gain * dxy[1],
            self.desired_z,
        ])

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
            self.context,
            JacobianWrtVariable.kV,
            self.gripper_body.body_frame(),
            [0, 0, 0],
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        tau_full = J_WG.T @ wrench_W
        tau_full -= self.plant.CalcGravityGeneralizedForces(self.context)
        output.set_value(tau_full[:7])
