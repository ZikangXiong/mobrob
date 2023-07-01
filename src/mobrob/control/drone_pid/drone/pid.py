import numpy as np
import pybullet as p
from scipy.optimize import nnls

from mobrob.envs.pybullet_robots.robots.drone import Drone

from ..base import ControllerBase


class DronePID(ControllerBase):
    """
    Adapted from
    https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/control/SimplePIDControl.py
    """

    def __init__(self, drone: Drone):
        self.drone = drone

        # Default PID coefficients
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
