import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    Context,
    AbstractValue,
    ImageDepth32F,
)

class Controller(LeafSystem):
    """PID controller for the IIWA robot"""

    def __init__(self, q_desired: np.ndarray) -> None:
        LeafSystem.__init__(self)

        self.input_port = self.DeclareVectorInputPort("iiwa_state", 14)
        self.output_port = self.DeclareVectorOutputPort(
            "iiwa_torque", 7, self.ComputeTorque
        )

        self.kp = 100
        self.kd = 100
        self.ki = 100
        self.q_desired = q_desired[:7]
        self.qdot_desired = np.zeros(7)
        self.integral_error = np.zeros(7)

        self.prev_time = 0.0

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        # TODO: Extract state information (same as PD controller)
        iiwa_state = self.input_port.Eval(context)
        q = iiwa_state[:7]  # YOUR CODE HERE
        qdot = iiwa_state[7:]  # YOUR CODE HERE

        current_time = context.get_time()
        dt = current_time - self.prev_time

        # TODO: Compute position and velocity errors (same as PD controller)
        position_error = self.q_desired - q
        velocity_error = self.qdot_desired - qdot

        # TODO: Update integral error
        if dt > 0:  # Avoid division by zero on first call
            self.integral_error += dt * position_error

        # TODO: Compute PID control law
        # HINT: Combine all three terms: proportional + derivative + integral
        torque = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        # Update previous time for next iteration
        self.prev_time = current_time

        output.set_value(torque)

class DepthController(LeafSystem):
    def __init__(self, plant) -> None:
        super().__init__()

        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()

        self.alpha = 1.0
        self.beta = 1.0

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
        # contact term
        f_contact = self.contact_port.Eval(context)
        tau_contact = -self.alpha * f_contact

        # gradient term - TODO
        depth_img: ImageDepth32F = self.depth_port.Eval(context)
        depth = np.array(depth_img.data, copy=False)[:, :, 0]

        valid = np.isfinite(depth)
        if np.any(valid):
            gy, gx = np.gradient(depth)
            H, W = depth.shape
            cy, cx = H // 2, W // 2
            grad_phi = np.array([gx[cy, cx], gy[cy, cx]])

            tau_grad = -self.beta * np.array(
                [grad_phi[0], grad_phi[1], 0, 0, 0, 0, 0]
            )
        else:
            tau_grad = np.zeros(7)

        # gravity term
        tau_g_full = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        tau_g = tau_g_full[:7]

        tau = tau_contact + tau_grad - tau_g
        output.set_value(tau)