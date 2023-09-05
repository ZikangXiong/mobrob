import numpy as np
import pybullet as p
from mobrob.envs.pybullet_robots.base import RobotBase
from mobrob.envs.pybullet_robots.utils import ROBOT_ASSETS_PATH
from mobrob.envs.pybullet_robots.worlds.turtlebot3 import World
from mobrob.utils import suppress_stdout


class Turtlebot3(RobotBase):
    def __init__(
        self,
        ray_length: float = 1.0,
        n_rays: int = 36,
        raycast_height_offset: float = 0.15,
        enable_gui: bool = False,
        plot_ray: bool = False,
    ):
        self.ray_length = ray_length
        self.n_rays = n_rays
        self.raycast_height_offset = raycast_height_offset
        self.enable_gui = enable_gui
        self.plot_ray = plot_ray

        super(Turtlebot3, self).__init__(World(enable_gui=enable_gui))

        self.ray_ids = []
        self.prop_gains = self.prop_gain_means

    def _init_param(self):
        # clock-wise ray angles
        increment = 2 * np.pi / self.n_rays
        self.ray_angles = np.array(
            [np.pi / 2 - increment * i for i in range(self.n_rays)]
        )
        self.default_pos, self.default_ori = (1.2, 1.2, 0), (0, 0, 0, 1)

        self.max_linear_vel = 0.26
        self.max_angular_vel = 1.82
        self.max_rpm = 100.0
        self.velocity_gain = 0.223
        self.force = 1

        # for twist control
        self.twist_l = 0.21  # meters
        self.twist_r = 0.032  # meters

        # for prop control
        self.prop_gain_means = np.array([1.0, 0.2])
        self.prop_gain_radius = (
            np.array([1.5, 0.5]) * 2
        )  # give the ability to go in reverse direction

    def set_pos_and_ori(self, pos: np.ndarray = None, ori: np.ndarray = None):
        pos = pos if pos is not None else self.default_pos
        ori = ori if ori is not None else self.default_ori

        if len(pos) == 2:
            z = p.getBasePositionAndOrientation(self.robot_id, self.client_id)[0][2]
            pos = np.r_[pos, z]
        p.resetBasePositionAndOrientation(
            self.robot_id, pos, ori, physicsClientId=self.client_id
        )

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        upper_action = np.r_[self.max_rpm, self.max_rpm]
        action = action.clip(min=-upper_action, max=upper_action)

        p.setJointMotorControl2(
            self.robot_id,
            1,
            p.VELOCITY_CONTROL,
            targetVelocity=action[0],
            physicsClientId=self.client_id,
            force=self.force,
            velocityGain=self.velocity_gain,
        )
        p.setJointMotorControl2(
            self.robot_id,
            2,
            p.VELOCITY_CONTROL,
            targetVelocity=action[1],
            physicsClientId=self.client_id,
            force=self.force,
            velocityGain=self.velocity_gain,
        )
        p.stepSimulation(self.client_id)

        obs = self.get_obs()

        return obs

    def zero_velocity(self):
        p.resetBaseVelocity(self.robot_id, self.client_id)

    def apply_action_twist(self, action: np.ndarray) -> np.ndarray:
        upper_action = np.r_[self.max_linear_vel, self.max_angular_vel]
        linear_vel, angular_vel = action.clip(min=-upper_action, max=upper_action)

        # calculate wheel velocities from target linear and angular
        left = (linear_vel / self.twist_r) + (angular_vel * self.twist_l / self.twist_r)
        right = (linear_vel / self.twist_r) - (
            angular_vel * self.twist_l / self.twist_r
        )

        return self.apply_action(np.array([left, right]))

    def get_obs(self) -> np.ndarray:
        state_dict = self._get_state_dict()
        state_flatten = []
        sorted_keys = ["theta", "x", "y", "linear_velocity", "angular_velocity"]
        for k in sorted_keys:
            if k == "linear_velocity":
                state_flatten.extend(state_dict[k][:2])
            elif k == "angular_velocity":
                state_flatten.extend(state_dict[k][2:])
            elif k == "theta":
                state_flatten.extend([np.sin(state_dict[k]), np.cos(state_dict[k])])
            else:
                state_flatten.append(state_dict[k])

        ray_obs = self._get_ray_obs()

        return np.concatenate([state_flatten, ray_obs])

    def get_pos(self) -> np.ndarray:
        return np.array(
            p.getBasePositionAndOrientation(self.robot_id, self.client_id)[0][:2]
        )

    def _load_robot(self) -> int:
        with suppress_stdout():
            robot_id = p.loadURDF(
                f"{ROBOT_ASSETS_PATH}/turtlebot3/turtlebot3_waffle.urdf",
                [1.2, 1.2, 0],
                physicsClientId=self.client_id,
            )
        return robot_id

    def _get_state_dict(self) -> dict:
        states = p.getLinkState(
            self.robot_id, 0, computeLinkVelocity=1, physicsClientId=self.client_id
        )
        current_local_frame_pos = states[0]
        current_local_frame_orient = p.getEulerFromQuaternion(states[1])
        robot_state_dict = {
            "x": current_local_frame_pos[0],
            "y": current_local_frame_pos[1],
            "theta": current_local_frame_orient[2],
            "linear_velocity": states[6],
            "angular_velocity": states[7],
        }
        return robot_state_dict

    def _get_ray_obs(self) -> np.ndarray:
        # robot state
        robot_state = p.getLinkState(self.robot_id, 0, physicsClientId=self.client_id)
        robot_angle = p.getEulerFromQuaternion(
            robot_state[1], physicsClientId=self.client_id
        )[2]
        robot_position = np.array(robot_state[0])

        # lidar scanned once
        raycast_pos = robot_position.copy()
        raycast_pos[2] += self.raycast_height_offset
        relative_angles = self.ray_angles - robot_angle
        relative_rays = self.ray_length * np.stack(
            [np.sin(relative_angles), np.cos(relative_angles), np.zeros(self.n_rays)],
            axis=1,
        )
        rays = raycast_pos + relative_rays
        results = p.rayTestBatch(
            [raycast_pos] * self.n_rays, rays, physicsClientId=self.client_id
        )

        # check lidar results
        obs = []
        for i, res in enumerate(results):
            if res[0] == -1:
                color = [0, 1, 0]
                hit_position = rays[i]
                dist = self.ray_length
            else:
                color = [1, 0, 0]
                hit_position = res[3]
                dist = np.linalg.norm(raycast_pos - np.array(hit_position))

            obs.append(dist)

            # visualize lidar
            if self.enable_gui and self.plot_ray:
                if len(self.ray_ids) == self.n_rays:
                    p.addUserDebugLine(
                        raycast_pos,
                        hit_position,
                        color,
                        replaceItemUniqueId=self.ray_ids[i],
                        physicsClientId=self.client_id,
                    )
                else:
                    self.ray_ids.append(
                        p.addUserDebugLine(
                            robot_position,
                            hit_position,
                            color,
                            physicsClientId=self.client_id,
                        )
                    )

        return np.array(obs)

    def prop_ctrl(self, pos_goal: np.ndarray, gain_changes: np.ndarray) -> np.ndarray:
        prop_gains = self.prop_gain_means + self.prop_gain_radius * gain_changes
        state_dict = self._get_state_dict()
        pos = np.array([state_dict["x"], state_dict["y"]])
        goal_vec = pos_goal - pos

        dist_prop = np.linalg.norm(goal_vec)

        angle_goal = np.arccos(
            np.dot(goal_vec, np.array([1.0, 0])) / (dist_prop + 1e-5)
        ) * np.sign(goal_vec[1])

        angle_prop = -(angle_goal - state_dict["theta"])
        if angle_prop > np.pi:
            angle_prop -= 2 * np.pi
        elif angle_prop < -np.pi:
            angle_prop += 2 * np.pi

        twist_cmd = np.array([dist_prop, angle_prop]) * prop_gains

        cmd_high = np.array([self.max_linear_vel, self.max_angular_vel])
        cmd_low = -cmd_high
        twist_cmd = twist_cmd.clip(cmd_low, cmd_high)

        return twist_cmd

    def reset(self, init_pos: np.ndarray = None):
        if init_pos is not None:
            p.resetBasePositionAndOrientation(
                self.robot_id,
                init_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                self.client_id,
            )

        self.zero_velocity()

    def ctrl(self, cmd: np.ndarray):
        self.apply_action_twist(cmd)
