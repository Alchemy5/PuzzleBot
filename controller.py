import numpy as np
from pydrake.all import LeafSystem, BasicVector, Context
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