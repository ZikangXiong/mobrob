from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.wrappers import TimeLimit

from mobrob.envs.mujoco_robots.robots.engine import Engine, quat2zalign
from mobrob.envs.pybullet_robots.base import BulletEnv
from mobrob.envs.pybullet_robots.robots.drone import Drone
from mobrob.envs.pybullet_robots.robots.turtlebot3 import Turtlebot3


class EnvWrapper(ABC, gym.Env):
    def __init__(
        self,
        enable_gui: bool = False,
        terminate_on_goal: bool = False,
    ):
        """
        Gym environment wrapper for robots
        :param enable_gui: whether to enable the GUI
        """
        self.enable_gui = enable_gui
        self.terminate_on_goal = terminate_on_goal
        self._goal = None
        self._prev_pos = None

        self.env: Engine | BulletEnv = self.build_env()
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        self.init_space = self.get_init_space()
        self.goal_space = self.get_goal_space()

        self._first_reset = True
        self.render_mode = "human"

    def seed(self, seed=None):
        """
        Seed the environment
        """
        self.env.seed(seed)

        # seed the spaces
        self.init_space.seed(seed)
        self.goal_space.seed(
            seed + 1 if seed is not None else None
        )  # avoid init on goal
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def toggle_render_mode(self):
        """
        Toggle the render mode between "human" and "rgb_array"
        """
        self.render_mode = "human" if self.render_mode == "rgb_array" else "rgb_array"

    @abstractmethod
    def _set_goal(self, goal: list | np.ndarray):
        """
        Set the goal position of the robot, for example, [x, y, z]
        """
        raise NotImplementedError()

    @abstractmethod
    def build_env(self) -> Engine | BulletEnv:
        """
        Build the environment, for example, load the robot and the world
        """
        raise NotImplementedError()

    @abstractmethod
    def get_pos(self):
        """
        Get the position of the robot, for example, [x, y, z]
        """
        pass

    @abstractmethod
    def set_pos(self, pos: list | np.ndarray):
        """
        Set the position of the robot, for example, [x, y, z]
        """
        pass

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """
        Get the observation of the robot, for example, [x, y, z]
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        """
        Get the observation space of the robot, for example, Box(3,)
        """
        pass

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """
        Get the action space of the robot, for example, Box(3,)
        """
        pass

    @abstractmethod
    def get_init_space(self) -> gym.Space:
        """
        Get the init space of the robot, for example, Box(3,)
        """
        pass

    @abstractmethod
    def get_goal_space(self) -> gym.Space:
        """
        Get the goal space of the robot, for example, Box(3,)
        """
        pass

    def set_goal(self, goal: list | np.ndarray):
        """
        Set the goal position of the robot, for example, [x, y, z]
        """
        self._set_goal(goal)
        self._goal = np.array(goal)

    def reset_random_goal(self):
        """
        Reset the goal position of the robot randomly
        """
        self.set_goal(self.goal_space.sample())

    def get_goal(self) -> np.ndarray:
        """
        Get the goal position of the robot, for example, [x, y, z]
        """
        return self._goal

    def reward_fn(self) -> float:
        """
        The default reward is the time derivative of the distance to the goal
        Overwrite this function if you want to use a different reward function
        """
        current_pos = self.get_pos()
        if self._goal is None or self._prev_pos is None:
            reward = 0
        else:
            reward = np.linalg.norm(self._goal - self._prev_pos) - np.linalg.norm(
                self._goal - current_pos
            )
        self._prev_pos = current_pos

        if self.reached():
            reward += 5.0

        return reward

    def step(
        self, action: list | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment by applying the action to the robot,
        the returns are the observation, reward, terminated, trucated, info
        """

        obs, _, terminated, trucated, info = self.env.step(action)

        reward = self.reward_fn()

        # this makes the value function simpler as it avoids the randomness introduced by the auto reset goal
        terminated = self.terminate_on_goal and self.reached()

        return obs, reward, terminated, trucated, info

    def reset(
        self, init_pos: list | np.ndarray = None, *args, **kwargs
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment, the return is the observation and reset info
        """
        if "seed" in kwargs:
            self.seed(kwargs.pop("seed"))

        if self._first_reset or not self.reached():
            # Only reset the robot if it has not reached the goal, this saves lots of simulation time.
            # During training, calling reset when the robot does not meet the goal means the time limit is reached.
            # To avoid situations such as getting stuck in a corner, we do not reset the robot.
            # One corner case is that the time limit is reached while the robot also reaches the goal.
            # In this case, we do not reset the robot, because usually the robot is not stuck in a corner
            # if it can reach the goal.
            # The first reset is guaranteed to be called when the environment is created.
            self.env.reset()
            self.set_pos(self.init_space.sample())

        if init_pos is not None:
            self.set_pos(init_pos)

        self.reset_random_goal()
        self._prev_pos = self.get_pos()

        self._first_reset = False

        return self.get_obs(), {}

    def reached(self, reach_radius: float = 0.3) -> bool:
        """
        Check if the robot has reached the goal
        """
        return np.linalg.norm(self.get_pos() - self.get_goal()) < reach_radius

    def render(self):
        """
        Render the environment
        """
        return self.env.render(mode=self.render_mode)

    def close(self):
        self.env.close()


class MujocoGoalEnv(EnvWrapper, ABC):
    BASE_SENSORS = ["accelerometer", "velocimeter", "gyro", "magnetometer"]

    @abstractmethod
    def get_robot_config(self) -> dict:
        pass

    def build_env(self) -> Engine:
        config = self.get_robot_config()
        env = Engine(config)

        return env

    def get_observation_space(self) -> gym.Space:
        return self.env.observation_space

    def get_action_space(self) -> gym.Space:
        return self.env.action_space

    def get_init_space(self) -> gym.Space:
        x_min, y_min, x_max, y_max = self.env.placements_extents
        return gym.spaces.Box(
            low=np.array([x_min, y_min], dtype=np.float32) / 2,
            high=np.array([x_max, y_max], dtype=np.float32) / 2,
            dtype=np.float32,
        )

    def get_goal_space(self) -> gym.Space:
        x_min, y_min, x_max, y_max = self.env.placements_extents
        return gym.spaces.Box(
            low=np.array([x_min, y_min], dtype=np.float32),
            high=np.array([x_max, y_max], dtype=np.float32),
            dtype=np.float32,
        )

    def _set_goal(self, goal: list | np.ndarray):
        self.env.set_goal_position(goal_xy=goal[:2])

    def get_pos(self) -> np.ndarray:
        return np.array(self.env.robot_pos[:2])

    def get_obs(self) -> np.ndarray:
        return self.env.obs()

    def add_wp_marker(
        self,
        pos: list | np.ndarray,
        size: float,
        color=(0, 1, 1, 0.5),
        alpha=0.5,
        label: str = "",
    ):
        self.env.add_render_callback(
            lambda: self.env.render_sphere(
                pos=pos, size=size, color=color, alpha=alpha, label=label
            )
        )


class PointEnv(MujocoGoalEnv):
    render_mode = "rgb_array"

    def get_robot_config(self) -> dict:
        return {
            "robot_base": f"xmls/point.xml",
            "sensors_obs": self.BASE_SENSORS,
            "observe_com": False,
            "observe_goal_comp": True,
        }

    def set_pos(self, pos: list | np.ndarray):
        body_id = self.env.sim.model.body_name2id("robot")
        self.env.sim.model.body_pos[body_id][:2] = pos
        self.env.sim.data.body_xpos[body_id][:2] = pos
        self.env.sim.forward()


class CarEnv(MujocoGoalEnv):
    def get_robot_config(self) -> dict:
        return {
            "robot_base": f"xmls/car.xml",
            "sensors_obs": self.BASE_SENSORS,
            "observe_com": False,
            "observe_goal_comp": True,
            "box_size": 0.125,  # Box half-radius size
            "box_keepout": 0.125,  # Box keepout radius for placement
            "box_density": 0.0005,
        }

    def set_pos(self, pos: list | np.ndarray):
        indx = self.env.sim.model.get_joint_qpos_addr("robot")
        sim_state = self.env.sim.get_state()

        sim_state.qpos[indx[0] : indx[0] + 2] = pos
        self.env.sim.set_state(sim_state)
        self.env.sim.forward()


class DoggoEnv(MujocoGoalEnv):
    def get_robot_config(self) -> dict:
        extra_sensor = [
            "touch_ankle_1a",
            "touch_ankle_2a",
            "touch_ankle_3a",
            "touch_ankle_4a",
            "touch_ankle_1b",
            "touch_ankle_2b",
            "touch_ankle_3b",
            "touch_ankle_4b",
        ]
        return {
            "robot_base": f"xmls/doggo.xml",
            "sensors_obs": self.BASE_SENSORS + extra_sensor,
            "observe_com": False,
            "observe_goal_comp": True,
        }

    def reward_fn(self) -> float:
        get_closer_r = super().reward_fn()
        zalign = quat2zalign(self.env.data.get_body_xquat("robot"))
        upright_r = 0.002 * zalign

        return get_closer_r + upright_r

    def set_pos(self, pos: list | np.ndarray):
        indx = self.env.sim.model.get_joint_qpos_addr("robot")
        sim_state = self.env.sim.get_state()

        sim_state.qpos[indx[0] : indx[0] + 2] = pos
        self.env.sim.set_state(sim_state)
        self.env.sim.forward()


class BulletGoalEnv(EnvWrapper, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.goal_shape_id = None
        self.goal_id = None
        self.reach_radius = 0.3

    def _set_goal(self, goal: list | np.ndarray):
        self._prev_pos = None

        if self.enable_gui:
            self.render_goal(goal)

    def render_goal(self, goal: np.ndarray):
        if self.enable_gui:
            if len(goal) == 2:
                goal = np.r_[goal, 0]
            if self.goal_shape_id is None:
                self.goal_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=self.reach_radius,
                    rgbaColor=[0, 1, 0, 0.5],
                    specularColor=[0.4, 0.4, 0],
                    physicsClientId=self.env.client_id,
                )
                self.goal_id = p.createMultiBody(
                    baseVisualShapeIndex=self.goal_shape_id,
                    basePosition=goal,
                    useMaximalCoordinates=True,
                    physicsClientId=self.env.client_id,
                )
            else:
                p.resetBasePositionAndOrientation(
                    self.goal_id, goal, [0, 0, 0, 1], physicsClientId=self.env.client_id
                )


class DroneEnv(BulletGoalEnv):
    def build_env(self) -> gym.Env:
        return BulletEnv(Drone(enable_gui=self.enable_gui))

    def get_pos(self) -> np.ndarray:
        return np.array(
            p.getBasePositionAndOrientation(
                self.env.robot.robot_id, self.env.client_id
            )[0]
        )

    def set_pos(self, pos: list | np.ndarray):
        p.resetBasePositionAndOrientation(
            self.env.robot.robot_id, pos, [0, 0, 0, 1], self.env.client_id
        )

    def get_obs(self) -> np.ndarray:
        obs = self.env.get_obs()
        obs[0:3] = obs[0:3] - self.get_goal()

        return obs

    def get_observation_space(self) -> gym.Space:
        high = np.array(
            [
                # x, y, z
                10.0,
                10.0,
                5.0,
                # roll, pitch, yaw
                np.pi,
                np.pi,
                np.pi,
                # vx, vy, vz
                15.0,
                15.0,
                15.0,
                # roll rate, pitch rate, yaw rate
                0.2 * np.pi,
                0.2 * np.pi,
                0.2 * np.pi,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                # x, y, z
                -10.0,
                -10.0,
                -50.0,
                # roll, pitch, yaw
                -np.pi,
                -np.pi,
                -np.pi,
                # vx, vy, vz
                -15.0,
                -15.0,
                -15.0,
                # roll rate, pitch rate, yaw rate
                -0.2 * np.pi,
                -0.2 * np.pi,
                -0.2 * np.pi,
            ],
            dtype=np.float32,
        )
        return gym.spaces.Box(low=low, high=high, shape=(12,), dtype=np.float32)

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(18,), dtype=np.float32)

    def get_init_space(self) -> gym.Space:
        lb = np.array([-5, -5, 5], dtype=np.float32)
        ub = np.array([5, 5, 10], dtype=np.float32)
        return gym.spaces.Box(low=lb, high=ub, dtype=np.float32)

    def get_goal_space(self) -> gym.Space:
        lb = np.array([-5, -5, 0], dtype=np.float32)
        ub = np.array([5, 5, 5], dtype=np.float32)
        return gym.spaces.Box(low=lb, high=ub, dtype=np.float32)

    def step(
        self, action: list | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = action.reshape((6, 3))
        self.env.robot.controller.finetune_force_pid_coef(*action[:3])
        self.env.robot.controller.finetune_torque_pid_coef(*action[3:])
        action = self.env.robot.controller.control(self.get_goal())

        return super().step(action)

    def reward_fn(self) -> float:
        super_reward = super().reward_fn()
        if self.reached():
            # drone moves fast, the reach composition should be larger
            return super_reward + 10.0
        return super_reward


class Turtlebot3Env(BulletGoalEnv):
    def build_env(self) -> Engine | BulletEnv:
        return BulletEnv(Turtlebot3(enable_gui=self.enable_gui))

    def get_pos(self):
        return self.env.robot.get_pos()

    def set_pos(self, pos: list | np.ndarray):
        self.env.robot.set_pos_and_ori(pos, None)

    def get_obs(self) -> np.ndarray:
        obs = self.env.get_obs()
        obs[2:4] = obs[2:4] - self.get_goal()

        return obs

    def get_observation_space(self) -> gym.Space:
        upper_x, upper_y, upper_sin_cos = 1.0, 1.0, 1.0
        ray_length = self.env.robot.ray_length
        upper_lin_vel = self.env.robot.max_linear_vel
        upper_angle_vel = self.env.robot.max_angular_vel

        max_dist = (upper_x**2 + upper_y**2) ** 0.5
        upper_obs = [upper_sin_cos, upper_sin_cos, max_dist, max_dist]
        upper_obs += [upper_lin_vel, upper_lin_vel, upper_angle_vel]
        upper_obs += [ray_length] * self.env.robot.n_rays

        upper_obs = np.array(upper_obs, dtype=np.float32)
        observation_space = gym.spaces.Box(low=-upper_obs, high=upper_obs)

        return observation_space

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def get_init_space(self) -> gym.Space:
        return gym.spaces.Box(low=-0.8, high=0.8, shape=(2,), dtype=np.float32)

    def get_goal_space(self) -> gym.Space:
        return gym.spaces.Box(low=-0.8, high=0.8, shape=(2,), dtype=np.float32)

    def step(
        self, action: list | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        gain_changes = np.array(action, dtype=np.float32)
        twist_cmd = self.env.robot.prop_ctrl(self.get_goal(), gain_changes)

        return super().step(twist_cmd)


def get_env(
    env_name: str,
    enable_gui: bool = False,
    terminate_on_goal: bool = False,
    time_limit: int | None = None,
):
    if env_name == "drone":
        env = DroneEnv(enable_gui, terminate_on_goal)
    elif env_name == "point":
        env = PointEnv(enable_gui, terminate_on_goal)
    elif env_name == "car":
        env = CarEnv(enable_gui, terminate_on_goal)
    elif env_name == "doggo":
        env = DoggoEnv(enable_gui, terminate_on_goal)
    elif env_name == "turtlebot3":
        env = Turtlebot3Env(enable_gui, terminate_on_goal)
    else:
        raise ValueError(f"Env {env_name} not found")

    if time_limit is not None:
        env = TimeLimit(env, max_episode_steps=time_limit)

    return env
