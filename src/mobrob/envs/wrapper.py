from abc import ABC, abstractmethod
from typing import List, Union

import gym
import numpy as np

from mobrob.envs.mujoco_robots.robots.engine import Engine
from mobrob.envs.pybullet_robots.base import BulletEnv
from mobrob.envs.pybullet_robots.robots.drone import Drone
from mobrob.stl.tasks import TaskBase


class EnvWrapper(ABC):
    def __init__(self, task: TaskBase, enable_gui: bool = True):
        self.task = task
        self.enable_gui = enable_gui
        self.obstacle_list = self.task.task_map.obs_list
        self.wp_list = self.task.wp_list
        self._goal = None
        self.gym_env: Union[Engine, BulletEnv] = self.build_env()

    @abstractmethod
    def _set_goal(self, goal: Union[List, np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def build_env(self) -> Union[Engine, BulletEnv]:
        raise NotImplementedError()

    @abstractmethod
    def get_pos(self):
        pass

    @abstractmethod
    def set_pos(self, pos: Union[List, np.ndarray]):
        pass

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass

    def set_goal(self, goal: Union[List, np.ndarray]):
        self._set_goal(goal)
        self._goal = np.array(goal)

    def get_goal(self) -> np.ndarray:
        return self._goal

    def step(self, action: Union[List, np.ndarray]):
        if self.enable_gui:
            self.gym_env.render()
        return self.gym_env.step(action)

    def reset(self, init_pos: Union[List, np.ndarray] = None):
        self.gym_env.reset()
        if init_pos is not None:
            self.set_pos(init_pos)
        return self.get_obs()

    def reached(self, reach_radius: float = 0.3) -> bool:
        return np.linalg.norm(self.get_pos() - self.get_goal()) < reach_radius

    def close(self):
        self.gym_env.close()


class MujocoEnv(EnvWrapper, ABC):
    BASE_SENSORS = ["accelerometer", "velocimeter", "gyro", "magnetometer"]

    def __init__(self, task: TaskBase, enable_gui: bool = True):
        super().__init__(task, enable_gui)
        for wp in self.task.wp_list:
            self.add_wp_marker(wp.pos, wp.size)

    def get_obs_config(self) -> dict:
        config = {
            "walls_num": len(self.task.task_map.obs_list),
            "walls_locations": [obs.pos for obs in self.task.task_map.obs_list],
            "walls_size": [obs.size for obs in self.task.task_map.obs_list],
        }
        return config

    @abstractmethod
    def get_robot_config(self) -> dict:
        pass

    def add_wp_marker(
        self,
        pos: Union[List, np.ndarray],
        size: float,
        color=(0, 1, 1, 0.5),
        alpha=0.5,
        label: str = "",
    ):
        self.gym_env.add_render_callback(
            lambda: self.gym_env.render_sphere(
                pos=pos, size=size, color=color, alpha=alpha, label=label
            )
        )

    def build_env(self) -> Engine:
        config = self.get_robot_config()
        config.update(self.get_obs_config())
        gym_env = Engine(config)

        return gym_env

    def _set_goal(self, goal: Union[List, np.ndarray]):
        self.gym_env.set_goal_position(goal_xy=goal[:2])

    def get_pos(self) -> np.ndarray:
        return np.array(self.gym_env.robot_pos[:2])

    def get_obs(self) -> np.ndarray:
        return self.gym_env.obs()


class PointEnv(MujocoEnv):
    def get_robot_config(self) -> dict:
        return {
            "robot_base": f"xmls/point.xml",
            "sensors_obs": self.BASE_SENSORS,
            "observe_com": False,
            "observe_goal_comp": True,
        }

    def set_pos(self, pos: Union[List, np.ndarray]):
        body_id = self.gym_env.sim.model.body_name2id("robot")
        self.gym_env.sim.model.body_pos[body_id][:2] = pos
        self.gym_env.sim.data.body_xpos[body_id][:2] = pos
        self.gym_env.sim.forward()


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

    def set_pos(self, pos: Union[List, np.ndarray]):
        indx = self.gym_env.sim.model.get_joint_qpos_addr("robot")
        sim_state = self.gym_env.sim.get_state()

        sim_state.qpos[indx[0] : indx[0] + 2] = pos
        self.gym_env.sim.set_state(sim_state)
        self.gym_env.sim.forward()


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

    def set_pos(self, pos: Union[List, np.ndarray]):
        indx = self.gym_env.sim.model.get_joint_qpos_addr("robot")
        sim_state = self.gym_env.sim.get_state()

        sim_state.qpos[indx[0] : indx[0] + 2] = pos
        self.gym_env.sim.set_state(sim_state)
        self.gym_env.sim.forward()


class DroneEnv(EnvWrapper):
    def build_env(self) -> gym.Env:
        return BulletEnv(Drone(enable_gui=self.enable_gui))

    def _set_goal(self, goal: Union[List, np.ndarray]):
        pass

    def get_pos(self) -> np.ndarray:
        # np.array(p.getBasePositionAndOrientation(self.robot_id, self.client_id)[0])
        pass

    def set_pos(self, pos: Union[List, np.ndarray]):
        pass

    def get_obs(self) -> np.ndarray:
        pass


def get_env(robot_name: str, task: TaskBase, enable_gui: bool = True):
    if robot_name == "drone":
        return DroneEnv(task, enable_gui)
    elif robot_name == "point":
        return PointEnv(task, enable_gui)
    elif robot_name == "car":
        return CarEnv(task, enable_gui)
    elif robot_name == "doggo":
        return DoggoEnv(task, enable_gui)
    else:
        raise ValueError(f"Env {robot_name} not found")
