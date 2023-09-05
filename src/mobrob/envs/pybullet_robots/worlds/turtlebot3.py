import pybullet as p
import pybullet_data

from mobrob.envs.pybullet_robots.base import WorldBase
from mobrob.envs.pybullet_robots.utils import ROBOT_ASSETS_PATH

URDF_FOLDER = f"{ROBOT_ASSETS_PATH}/turtlebot3"


class World(WorldBase):
    def _init_param(self):
        self.g = 9.8
        self.timestep = 1 / 50

    def _build_world(self):
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.setGravity(0, 0, -self.g, physicsClientId=self.client_id)
        p.setTimeStep(self.timestep, physicsClientId=self.client_id)

        p.resetDebugVisualizerCamera(
            1.8, 90, -89, (1.2, 1.2, 0), physicsClientId=self.client_id
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)

    def reset(self):
        pass
