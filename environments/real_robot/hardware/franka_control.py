import numpy as np
import torch
import torchcontrol as toco
from polymetis import RobotInterface
from torchcontrol.transform import Rotation as R
from torchcontrol.utils import to_tensor


class ResolvedRateControl(toco.PolicyModule):
    """Resolved Rates Control --> End-Effector Control (dx, dy, dz, droll, dpitch, dyaw) via Joint Velocity Control"""

    def __init__(
        self,
        Kp: torch.Tensor,
        robot_model: torch.nn.Module,
        ignore_gravity: bool = True,
    ) -> None:
        """
        Initializes a Resolved Rates controller with the given P gains and robot model.

        :param Kp: P gains in joint space (7-DoF)
        :param robot_model: A robot model from torchcontrol.models
        :param ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize Modules --> Inverse Dynamics is necessary as it needs to be compensated for in output torques...
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )

        # Create LinearFeedback (P) Controller...
        self.p = toco.modules.feedback.LinearFeedback(Kp)

        # Reference End-Effector Velocity (dx, dy, dz, droll, dpitch, dyaw)
        self.ee_velocity_desired = torch.nn.Parameter(torch.zeros(6))

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute joint torques given desired EE velocity.

        :param state_dict: A dictionary containing robot states (joint positions, velocities, etc.)
        :return Dictionary containing joint torques.
        """
        # State Extraction
        joint_pos_current, joint_vel_current = (
            state_dict["joint_positions"],
            state_dict["joint_velocities"],
        )

        # Compute Target Joint Velocity via Resolved Rate Control...
        #   =>> Resolved Rate: joint_vel_desired = J.pinv() @ ee_vel_desired
        #                      >> Numerically stable --> torch.linalg.lstsq(J, ee_vel_desired).solution
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        joint_vel_desired = torch.linalg.lstsq(
            jacobian, self.ee_velocity_desired
        ).solution

        # Control Logic --> Compute P Torque (feedback) & Inverse Dynamics Torque (feedforward)
        torque_feedback = self.p(joint_vel_current, joint_vel_desired)
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class HybridJointImpedanceControl(toco.PolicyModule):
    """
    Impedance control in joint space, but with both fixed joint gains and adaptive operational space gains.
    """

    def __init__(
        self,
        joint_pos_current,
        kq,
        kqd,
        kx,
        kxd,
        robot_model: torch.nn.Module,
        ki=None,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in Cartesian space
            Kd: D gains in Cartesian space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(kq, kqd, kx, kxd)
        self.ki = ki

        self.kp = torch.nn.Parameter(torch.tensor(0.0))
        self.kd = torch.nn.Parameter(torch.tensor(0.0))
        # Reference pose
        self.q_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.qd_desired = torch.nn.Parameter(torch.zeros_like(joint_pos_current))

        self.integral_error = torch.zeros_like(joint_pos_current)

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states

        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        # print(state_dict)
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        if self.ki is not None:
            error = self.q_desired - joint_pos_current
            error_integral = (
                self.integral_error + error
            )  # Accumulate error for integral term
            self.integral_error = error_integral  # Update integral error

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.q_desired,
            self.qd_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward
        if self.ki is not None:
            torque_out += self.ki @ error_integral

        return {"joint_torques": torque_out}


class JointPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, joint_pos_current, kp, kd, **kwargs):
        """
        Args:
            desired_joint_pos (int):    Number of steps policy should execute
            hz (double):                Frequency of controller
            kp, kd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.kp = torch.nn.Parameter(kp)
        self.kd = torch.nn.Parameter(kd)
        self.q_desired = torch.nn.Parameter(joint_pos_current)
        self.qd_desired = torch.nn.Parameter(torch.zeros_like(joint_pos_current))

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(self.kp, self.kd)

    def forward(self, state_dict: dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]
        self.feedback.Kp = torch.diag(self.kp)
        self.feedback.Kd = torch.diag(self.kd)

        # Execute PD control
        output = self.feedback(q_current, qd_current, self.q_desired, self.qd_desired)
        return {"joint_torques": output}


class HumanControl(toco.PolicyModule):
    # stolen from SimulationFramework
    def __init__(self, robot: RobotInterface, regularize=True):
        super().__init__()

        # get joint limits for regularization
        limits = robot.robot_model.get_joint_angle_limits()
        self.joint_pos_min = limits[0]
        self.joint_pos_max = limits[1]

        # define gain
        self.gain = torch.Tensor([0.3, 0.12, 0.40, 1.11, 1.10, 0.6, 0.85])

        if regularize:
            self.reg_gain = torch.Tensor([5.0, 2.2, 1.3, 0.3, 0.1, 0.1, 0.0])
        else:
            self.reg_gain = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def forward(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ext = state_dict["motor_torques_external"]

        human_torque = -self.gain * ext
        joint_pos_current = state_dict["joint_positions"]

        left_boundary = 1 / torch.clamp(
            torch.abs(self.joint_pos_min - joint_pos_current), 1e-8, 100000
        )
        right_boundary = 1 / torch.clamp(
            torch.abs(self.joint_pos_max - joint_pos_current), 1e-8, 100000
        )

        reg_load = left_boundary - right_boundary

        reg_torgue = self.reg_gain * reg_load

        return {"joint_torques": human_torque + reg_torgue}


class ImitationControl(HybridJointImpedanceControl):
    def __init__(self, robot: RobotInterface):
        super().__init__(
            joint_pos_current=robot.get_joint_positions(),
            kq=robot.Kq_default,
            kqd=robot.Kqd_default,
            kx=robot.Kx_default,
            kxd=robot.Kxd_default,
            robot_model=robot.robot_model,
            ignore_gravity=robot.use_grav_comp,
        )
        # super().__init__(
        #     joint_pos_current=robot.get_joint_positions(),
        #     kq=robot.Kq_default*0.85,
        #     kqd=robot.Kqd_default*0.5,
        #     kx=robot.Kx_default*0.35,
        #     kxd=robot.Kxd_default*0.25,
        #     ki=robot.Kq_default*0.001,
        #     robot_model=robot.robot_model,
        #     ignore_gravity=robot.use_grav_comp,
        # )
