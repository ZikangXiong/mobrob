from xml.etree import ElementTree

import numpy as np
import pybullet as p
from scipy.optimize import nnls

from mobrob.envs.pybullet_robots.base import RobotBase
from mobrob.envs.pybullet_robots.utils import ROBOT_ASSETS_PATH
from mobrob.envs.pybullet_robots.worlds.drone import World


class DronePIDController:
    """
    Adapted from
    https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/control/SimplePIDControl.py
    """

    def __init__(self, drone):
        self.drone = drone

        # Default PID coefficients
        self._force_p_coef_mean = np.array([0.1, 0.1, 0.2])
        self._force_i_coef_mean = np.array([0.0001, 0.0001, 0.0001])
        self._force_d_coef_mean = np.array([0.3, 0.3, 0.4])
        self._torque_p_coef_mean = np.array([0.3, 0.3, 0.05])
        self._torque_i_coef_mean = np.array([0.0001, 0.0001, 0.0001])
        self._torque_d_coef_mean = np.array([0.3, 0.3, 0.5])

        tune_fac = 0.3
        self._force_p_coef_r = np.array([0.1, 0.1, 0.2]) * tune_fac
        self._force_i_coef_r = np.array([0.0001, 0.0001, 0.0001]) * tune_fac
        self._force_d_coef_r = np.array([0.3, 0.3, 0.4]) * tune_fac
        self._torque_p_coef_r = np.array([0.3, 0.3, 0.05]) * tune_fac
        self._torque_i_coef_r = np.array([0.0001, 0.0001, 0.0001]) * tune_fac
        self._torque_d_coef_r = np.array([0.3, 0.3, 0.5]) * tune_fac

        self._force_p_coef = np.array([0.1, 0.1, 0.2])
        self._force_i_coef = np.array([0.0001, 0.0001, 0.0001])
        self._force_d_coef = np.array([0.3, 0.3, 0.4])
        self._torque_p_coef = np.array([0.3, 0.3, 0.05])
        self._torque_i_coef = np.array([0.0001, 0.0001, 0.0001])
        self._torque_d_coef = np.array([0.3, 0.3, 0.5])

        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        # constrain the max roll and pitch angles for stability
        self.max_roll_pitch = np.pi / 6

    def reset(self):
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def control(self, goal: np.ndarray) -> np.ndarray:
        cur_pos, cur_ori = self.drone.get_base_pos_and_ori()

        target_force = self._compute_target_force(goal, cur_pos)
        target_thrust = self._compute_target_thrust(target_force, cur_ori)

        target_rpy = self._compute_target_rotation(target_force)
        target_torque = self._compute_target_torque(target_rpy, cur_ori)

        # control commands in rpm
        rpm = self._compute_rpm(target_thrust, target_torque)

        return rpm

    def _compute_target_force(
        self, goal: np.ndarray, cur_pos: np.ndarray
    ) -> np.ndarray:
        pos_e = goal - np.array(cur_pos).reshape(3)
        d_pos_e = (pos_e - self.last_pos_e) / self.drone.world.timestep
        self.last_pos_e = pos_e
        self.integral_pos_e = self.integral_pos_e + pos_e * self.drone.world.timestep

        target_force = (
            np.array([0, 0, self.drone.m * self.drone.world.g])
            + np.multiply(self._force_p_coef, pos_e)
            + np.multiply(self._force_i_coef, self.integral_pos_e)
            + np.multiply(self._force_d_coef, d_pos_e)
        )

        return target_force

    def _compute_target_thrust(
        self, target_force: np.ndarray, cur_ori: np.ndarray
    ) -> float:
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_ori)).reshape(3, 3)
        target_thrust = np.dot(cur_rotation, target_force)
        target_thrust = target_thrust.clip(0, self.drone.max_thrust)

        return target_thrust[2]

    def _compute_target_rotation(self, target_force: np.ndarray) -> np.ndarray:
        target_rpy = np.zeros(3)

        sign_z = np.sign(target_force[2])
        if sign_z == 0:
            sign_z = 1

        target_rpy[0] = np.arcsin(
            -sign_z * target_force[1] / np.linalg.norm(target_force)
        )
        target_rpy[0] = np.clip(
            target_rpy[0], -self.max_roll_pitch, self.max_roll_pitch
        )
        target_rpy[1] = np.arctan2(sign_z * target_force[0], sign_z * target_force[2])
        target_rpy[1] = np.clip(
            target_rpy[1], -self.max_roll_pitch, self.max_roll_pitch
        )
        # Yaw is intended to leave as 0

        return target_rpy

    def _compute_target_torque(
        self, target_rpy: np.ndarray, cur_ori: np.ndarray
    ) -> np.ndarray:
        cur_rpy = p.getEulerFromQuaternion(cur_ori)
        rpy_e = target_rpy - np.array(cur_rpy)

        if rpy_e[2] > np.pi:
            rpy_e[2] = rpy_e[2] - 2 * np.pi
        if rpy_e[2] < -np.pi:
            rpy_e[2] = rpy_e[2] + 2 * np.pi
        d_rpy_e = (rpy_e - self.last_rpy_e) / self.drone.world.timestep
        self.last_rpy_e = rpy_e
        self.integral_rpy_e = self.integral_rpy_e + rpy_e * self.drone.world.timestep

        target_torque = (
            np.multiply(self._torque_p_coef, rpy_e)
            + np.multiply(self._torque_i_coef, self.integral_rpy_e)
            + np.multiply(self._torque_d_coef, d_rpy_e)
        )

        max_xy_torque = self.drone.max_xy_torque
        max_z_torque = self.drone.max_z_torque

        ub = np.array([max_xy_torque, max_xy_torque, max_z_torque])
        lb = -ub
        target_torque = target_torque.clip(lb, ub)

        return target_torque

    def _compute_rpm(
        self, target_thrust: float, target_torque: np.ndarray
    ) -> np.ndarray:
        x = np.concatenate([[target_thrust], target_torque])
        bx = self.drone.B * x
        power_rpm = self.drone.A_inv @ bx
        power_rpm = power_rpm.clip(0, self.drone.max_rpm**2)

        if np.min(power_rpm) < 0:
            power_rpm, res = nnls(self.drone.A, bx, maxiter=20)

        return np.sqrt(power_rpm)

    def set_force_pid_coef(
        self, p_coef: np.ndarray, i_coef: np.ndarray, d_coef: np.ndarray
    ):
        self._force_p_coef = p_coef
        self._force_i_coef = i_coef
        self._force_d_coef = d_coef

    def set_torque_pid_coef(
        self, p_coef: np.ndarray, i_coef: np.ndarray, d_coef: np.ndarray
    ):
        self._torque_p_coef = p_coef
        self._torque_i_coef = i_coef
        self._torque_d_coef = d_coef

    def finetune_force_pid_coef(
        self, p_coef_der: np.ndarray, i_coef_der: np.ndarray, d_coef_der: np.ndarray
    ):
        self._force_p_coef = self._force_p_coef_mean + p_coef_der * self._force_p_coef_r
        self._force_i_coef = self._force_i_coef_mean + i_coef_der * self._force_i_coef_r
        self._force_d_coef = self._force_d_coef_mean + d_coef_der * self._force_d_coef_r

    def finetune_torque_pid_coef(
        self, p_coef_der: np.ndarray, i_coef_der: np.ndarray, d_coef_der: np.ndarray
    ):
        self._torque_p_coef = (
            self._torque_p_coef_mean + p_coef_der * self._torque_p_coef_r
        )
        self._torque_i_coef = (
            self._torque_i_coef_mean + i_coef_der * self._torque_i_coef_r
        )
        self._torque_d_coef = (
            self._torque_d_coef_mean + d_coef_der * self._torque_d_coef_r
        )


class Drone(RobotBase):
    def __init__(self, drone_name: str = "hb", enable_gui=True):
        self.urdf_path = f"{ROBOT_ASSETS_PATH}/drone"
        self.drone_name = drone_name
        self.init_low = [-3, -3, 1]
        self.init_high = [3, 3, 3]

        world = World(enable_gui=enable_gui)
        super(Drone, self).__init__(world)

        self.controller = DronePIDController(self)

    def reset(self, init_pos: np.ndarray = None):
        if init_pos is not None:
            p.resetBasePositionAndOrientation(
                self.robot_id,
                init_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                self.client_id,
            )
        self.controller.reset()

    def _load_robot(self) -> int:
        robot_id = p.loadURDF(
            self.urdf_path + f"/{self.drone_name}.urdf",
            basePosition=[2, 2, 2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client_id,
        )

        return robot_id

    def _init_param(self):
        urdf_tree = ElementTree.parse(
            f"{self.urdf_path}/{self.drone_name}.urdf"
        ).getroot()

        self.m = float(urdf_tree[1][0][1].attrib["value"])
        self.l = float(urdf_tree[0].attrib["arm"])
        self.thrust2weight = float(urdf_tree[0].attrib["thrust2weight"])
        self.kf = float(urdf_tree[0].attrib["kf"])
        self.km = float(urdf_tree[0].attrib["km"])
        self.max_speed_kmh = float(urdf_tree[0].attrib["max_speed_kmh"])
        self.prop_radius = float(urdf_tree[0].attrib["prop_radius"])

        ixx = float(urdf_tree[1][0][2].attrib["ixx"])
        iyy = float(urdf_tree[1][0][2].attrib["iyy"])
        izz = float(urdf_tree[1][0][2].attrib["izz"])
        self.J = np.diag([ixx, iyy, izz])
        self.J_inv = np.linalg.inv(self.J)

        self.collision_h = float(urdf_tree[1][2][1][0].attrib["length"])
        self.collision_r = float(urdf_tree[1][2][1][0].attrib["radius"])
        collision_shape_offsets = [
            float(s) for s in urdf_tree[1][2][0].attrib["xyz"].split(" ")
        ]
        self.collision_z_offset = collision_shape_offsets[2]

        self.gnd_eff_coef = float(urdf_tree[0].attrib["gnd_eff_coeff"])

        drag_coeff_xy = float(urdf_tree[0].attrib["drag_coeff_xy"])
        drag_coeff_z = float(urdf_tree[0].attrib["drag_coeff_z"])
        self.drag_coef = np.array([drag_coeff_xy, drag_coeff_xy, drag_coeff_z])

        self.dw_coef_1 = float(urdf_tree[0].attrib["dw_coeff_1"])
        self.dw_coef_2 = float(urdf_tree[0].attrib["dw_coeff_2"])
        self.dw_coef_3 = float(urdf_tree[0].attrib["dw_coeff_3"])

        gravity = self.world.g * self.m  # noqa
        self.hover_rpm = np.sqrt(gravity / (4 * self.kf))
        self.max_rpm = np.sqrt((self.thrust2weight * gravity) / (4 * self.kf))

        self.max_thrust = 4 * self.kf * self.max_rpm**2
        if self.drone_name == "cf2x":
            self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm**2) / np.sqrt(2)
        elif self.drone_name in ["cf2p", "hb"]:
            self.max_xy_torque = self.l * self.kf * self.max_rpm**2

        self.max_z_torque = 2 * self.km * self.max_rpm**2

        # A and B are used for building the equation for computing control input
        self.A = np.array([[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
        self.A_inv = np.linalg.inv(self.A)
        self.B = np.array(
            [1 / self.kf, 1 / (self.kf * self.l), 1 / (self.kf * self.l), 1 / self.km]
        )

    def ctrl(self, cmd: np.ndarray):
        """
        Apply control cmd but do not step simulation
        :param cmd: rotation per minute of 4 motors
        """
        forces = np.array(cmd**2) * self.kf
        torques = np.array(cmd**2) * self.km
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for i in range(4):
            p.applyExternalForce(
                self.robot_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client_id,
            )
        p.applyExternalTorque(
            self.robot_id,
            -1,
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self.client_id,
        )

    def get_base_pos_and_ori(self) -> np.ndarray:
        return p.getBasePositionAndOrientation(self.robot_id, self.client_id)

    def get_obs(self) -> np.ndarray:
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, self.client_id)
        ori = p.getEulerFromQuaternion(ori)
        vel, rot = p.getBaseVelocity(self.robot_id, self.client_id)

        return np.concatenate([pos, ori, vel, rot])
