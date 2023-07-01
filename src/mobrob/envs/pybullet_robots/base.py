from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Union

import numpy as np
import pybullet as p

from mobrob.envs.pybullet_robots.utils import no_render

ObjID = namedtuple("ObjID", ["visual_id", "collision_id"])


class WorldBase(ABC):
    def __init__(self, enable_gui: bool):
        if enable_gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        self._init_param()

        with no_render():
            self._build_world()

        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

    @abstractmethod
    def _init_param(self):
        raise NotImplementedError()

    @abstractmethod
    def _build_world(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class RobotBase(ABC):
    def __init__(self, world: WorldBase):
        self.world = world

        self.client_id = self.world.client_id

        with no_render():
            self.robot_id = self._load_robot()

        self._init_param()

    @abstractmethod
    def _load_robot(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _init_param(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def ctrl(self, cmd: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass


class BulletEnv:
    def __init__(self, robot: RobotBase, camera_following: bool = False):
        self.world = robot.world
        self.robot = robot

        self._expose()
        self._camera_following = camera_following

    def _expose(self):
        self.client_id = self.world.client_id

    def switch_render_mode(self):
        self._camera_following = not self._camera_following

    def reset(self):
        self.world.reset()
        self.robot.reset()

    def get_obs(self):
        return self.robot.get_obs()

    def step(self, cmd: Union[np.ndarray, List[np.ndarray]]):
        self.robot.ctrl(cmd)
        p.stepSimulation(self.client_id)
        if self._camera_following:
            base_pos, base_orn = p.getBasePositionAndOrientation(
                self.robot.robot_id, physicsClientId=self.client_id
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=75,
                cameraPitch=-20,
                cameraTargetPosition=base_pos,
            )
        return self.get_obs(), 0, False, False, {}

    def render(self, mode="human"):
        pass
