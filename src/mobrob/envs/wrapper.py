from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.wrappers import TimeLimit

from mobrob.envs.mujoco_robots.robots.engine import Engine, quat2zalign
from mobrob.envs.pybullet_robots.base import BulletEnv
from mobrob.envs.pybullet_robots.robots.drone import Drone


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

    def seed(self, seed=None):
        """
        Seed the environment
        """
        self.env.seed(seed)

        # seed the spaces
        self.init_space.seed(seed)
        self.goal_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

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
        if self._goal is None:
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
        if self.enable_gui:
            self.env.render()

        obs, _, terminated, trucated, info = self.env.step(action)

        reward = self.reward_fn()

        # this makes the value function simpler as it avoids the randomness introduced by the auto reset goal
        terminated = self.terminate_on_goal and self.reached()

        return obs, reward, terminated, trucated, info

    def reset(self, init_pos: list | np.ndarray = None, *args, **kwargs) -> np.ndarray:
        """
        Reset the environment, the return is the observation and reset info
        """
        if "seed" in kwargs:
            self.seed(kwargs.pop("seed"))

        if len(args) > 0:
            print(f"Warning: args ({args}) is not empty, it is ignored.")
        if len(kwargs) > 0:
            print(f"Warning: kwargs ({kwargs}) is not empty, it is ignored.")

        if self._first_reset or not self.reached():
            # Only reset the robot if it has not reached the goal, this saves lots of simulation time.
            # Calling reset when the robot does not not meet the goal means the time limit is reached.
            # One corner case is that the time limit is reached while the robot also reaches the goal.
            # In this case, we also do not reset the robot.
            # However, the first reset is gurrenteed to be called when the environment is created.
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

    def render(self, mode="human"):
        """
        Render the environment
        """
        if self.enable_gui:
            return self.env.render(mode=mode)
        else:
            return None

    def close(self):
        self.env.close()


class MujocoEnv(EnvWrapper, ABC):
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


class PointEnv(MujocoEnv):
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


class CarEnv(MujocoEnv):
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


class DoggoEnv(MujocoEnv):
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


class DroneEnv(EnvWrapper):
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
        return self.env.get_obs()

    def get_observation_space(self) -> gym.Space:
        high = np.array(
            [
                # x, y, z
                20.0,
                20.0,
                20.0,
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
                -20.0,
                -20.0,
                0.0,
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
        return gym.spaces.Box(
            low=0, high=self.env.robot.max_rpm, shape=(4,), dtype=np.float32
        )

    def get_init_space(self) -> gym.Space:
        lb = np.array([-5, -5, 5], dtype=np.float32)
        ub = np.array([5, 5, 10], dtype=np.float32)
        return gym.spaces.Box(low=lb, high=ub, dtype=np.float32)

    def get_goal_space(self) -> gym.Space:
        lb = np.array([-20, -20, 20], dtype=np.float32)
        ub = np.array([-20, -20, 0], dtype=np.float32)
        return gym.spaces.Box(low=lb, high=ub, dtype=np.float32)

    def _set_goal(self, goal: list | np.ndarray):
        self._goal = goal
        self._prev_pos = None


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
    else:
        raise ValueError(f"Env {env_name} not found")

    if time_limit is not None:
        env = TimeLimit(env, max_episode_steps=time_limit)

    return env
