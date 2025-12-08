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
from pydrake.perception import PointCloud


import numpy as np
from pydrake.all import LeafSystem, BasicVector, RotationMatrix, JacobianWrtVariable


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
        self.hor_cloud_port = self.DeclareAbstractInputPort(
            "hor_cloud", AbstractValue.Make(PointCloud())
        )
        self.ver_cloud_port = self.DeclareAbstractInputPort(
            "ver_cloud", AbstractValue.Make(PointCloud())
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

    def perception_processing(hor_data, ver_data):
        pass

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        if self.qs is None:
            raise RuntimeError("initialize qs first")

        iiwa_state = self.state_port.Eval(context)

        # point cloud eval
        hor_data = self.hor_cloud_port.Eval(context)
        ver_data = self.ver_cloud_port.Eval(context)

        q = iiwa_state[:7]
        qdot = iiwa_state[7:]

        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        self.plant.SetVelocities(self.plant_context, self.iiwa, qdot)

        q_des = self.qs[self.idx]
        err_norm = np.linalg.norm(q_des - q)
        print(self.idx, err_norm)

        if err_norm < 0.01 and self.idx < len(self.qs) - 1:
            if self.idx == 1:
                self.wsg_ctrl.set_target(0)
            self.idx += 1
            q_des = self.qs[self.idx]
        elif err_norm < 0.01 and self.idx == len(self.qs) - 1:
            if not self.movement:
                print("Done! Holding final position")
                self.movement = True
                self.final_q = q.copy()  # Lock in the final position
            q_des = self.final_q  # Use the locked position, not current q

        current_time = context.get_time()
        dt = current_time - self.prev_time

        position_error = q_des - q
        velocity_error = self.qdot_desired - qdot

        if dt > 0:
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
